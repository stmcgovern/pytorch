# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DiTBlock(nn.Module):
    """Minimal DiT block: adaLN-Zero modulation + self-attention + MLP.

    Follows the architecture from "Scalable Diffusion Models with Transformers"
    (Peebles & Xie, 2023). The conditioning signal `c` drives adaptive
    layer-norm (adaLN) modulation of the attention and MLP sub-blocks.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp_w1 = nn.Linear(dim, 4 * dim)
        self.mlp_w2 = nn.Linear(4 * dim, dim)

        # adaLN-Zero: produces (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(c).unsqueeze(1)  # (B, 1, 6*dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Modulated self-attention
        h = self.norm1(x) * (1 + scale1) + shift1
        bsz, seq_len, _ = h.size()
        q = self.wq(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        x = x + gate1 * self.wo(attn)

        # Modulated MLP
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp_w2(F.gelu(self.mlp_w1(h)))
        return x


class VAEEncoder(nn.Module):
    """Minimal VAE encoder with GroupNorm."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=latent_dim)
        self.proj = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = F.silu(self.group_norm(h))
        return self.proj(h)


class DiTModel(nn.Module):
    """Minimal DiT: VAE encode -> patchify -> DiT blocks -> unpatchify."""

    def __init__(
        self,
        img_channels: int = 3,
        latent_dim: int = 16,
        dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        patch_size: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.vae_encoder = VAEEncoder(img_channels, latent_dim)
        self.patch_embed = nn.Conv2d(
            latent_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.unpatchify = nn.Linear(dim, latent_dim * patch_size * patch_size)

    def forward(
        self, images: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        z = self.vae_encoder(images)
        h = self.patch_embed(z)
        B, C, pH, pW = h.shape
        h = h.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        for block in self.blocks:
            h = block(h, cond)
        h = self.final_norm(h)
        h = self.unpatchify(h)  # (B, num_patches, latent_dim * P * P)
        return h


def _replicate_params(model, device_mesh):
    """Distribute all model parameters as Replicate on the given mesh."""
    for name, param in model.named_parameters():
        dist_param = nn.Parameter(
            distribute_tensor(param, device_mesh, [Replicate()])
        )
        parts = name.split(".")
        mod = model
        for part in parts[:-1]:
            mod = getattr(mod, part)
        setattr(mod, parts[-1], dist_param)


class DiTDTensorTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _create_model_and_inputs(self, device):
        torch.manual_seed(42)
        model = DiTModel(
            img_channels=3, latent_dim=16, dim=32, n_heads=4,
            n_layers=2, patch_size=2,
        ).to(device)
        B = 4
        images = torch.randn(B, 3, 8, 8, device=device)
        cond = torch.randn(B, 32, device=device)
        return model, images, cond

    @with_comms
    def test_dit_forward_batch_shard(self):
        """Batch-dim sharded forward matches replicated forward."""
        device_mesh = self.build_device_mesh()
        model, images, cond = self._create_model_and_inputs(self.device_type)

        model_ref = copy.deepcopy(model)
        with torch.no_grad():
            out_ref = model_ref(images, cond)

        _replicate_params(model, device_mesh)
        images_dt = distribute_tensor(images, device_mesh, [Shard(0)])
        cond_dt = distribute_tensor(cond, device_mesh, [Shard(0)])

        with torch.no_grad():
            out_dt = model(images_dt, cond_dt)

        self.assertEqual(out_dt.full_tensor(), out_ref)

    @with_comms
    def test_dit_backward_batch_shard(self):
        """Backward pass with batch-dim sharding produces correct gradients."""
        device_mesh = self.build_device_mesh()
        model, images, cond = self._create_model_and_inputs(self.device_type)
        model_ref = copy.deepcopy(model)

        out_ref = model_ref(images, cond)
        out_ref.sum().backward()

        _replicate_params(model, device_mesh)
        images_dt = distribute_tensor(
            images.detach().clone(), device_mesh, [Shard(0)]
        )
        cond_dt = distribute_tensor(
            cond.detach().clone(), device_mesh, [Shard(0)]
        )

        out_dt = model(images_dt, cond_dt)
        out_dt.sum().backward()

        for (name_ref, p_ref), (name_dt, p_dt) in zip(
            model_ref.named_parameters(), model.named_parameters()
        ):
            self.assertEqual(name_ref, name_dt)
            if p_ref.grad is not None:
                dt_grad = p_dt.grad
                if isinstance(dt_grad, DTensor):
                    dt_grad = dt_grad.full_tensor()
                # Partial-sum all-reduce introduces floating-point ordering
                # differences that accumulate through multiple layers.
                self.assertEqual(
                    dt_grad, p_ref.grad, atol=3e-2, rtol=3e-2,
                    msg=f"Gradient mismatch for {name_ref}",
                )

    @with_comms
    def test_dit_forward_no_comms_batch_shard(self):
        """Batch-dim sharding should not require any collective communication."""
        device_mesh = self.build_device_mesh()
        model, images, cond = self._create_model_and_inputs(self.device_type)

        _replicate_params(model, device_mesh)
        images_dt = distribute_tensor(images, device_mesh, [Shard(0)])
        cond_dt = distribute_tensor(cond, device_mesh, [Shard(0)])

        with CommDebugMode() as comm_mode:
            with torch.no_grad():
                model(images_dt, cond_dt)

        self.assertEqual(
            comm_mode.get_total_counts(), 0,
            "Expected zero collectives for batch-dim sharding",
        )

    @with_comms
    def test_group_norm_forward_batch_shard(self):
        """GroupNorm forward with batch-dim sharding matches replicated."""
        device_mesh = self.build_device_mesh()
        torch.manual_seed(0)
        gn = nn.GroupNorm(4, 16).to(self.device_type)
        x = torch.randn(4, 16, 8, 8, device=self.device_type)

        out_ref = gn(x)

        gn_dt = copy.deepcopy(gn)
        gn_dt.weight = nn.Parameter(
            distribute_tensor(gn_dt.weight, device_mesh, [Replicate()])
        )
        gn_dt.bias = nn.Parameter(
            distribute_tensor(gn_dt.bias, device_mesh, [Replicate()])
        )
        x_dt = distribute_tensor(x, device_mesh, [Shard(0)])

        out_dt = gn_dt(x_dt)
        self.assertEqual(out_dt.full_tensor(), out_ref)

    @with_comms
    def test_group_norm_backward_batch_shard(self):
        """GroupNorm backward with batch-dim sharding produces correct gradients."""
        device_mesh = self.build_device_mesh()
        torch.manual_seed(0)
        gn = nn.GroupNorm(4, 16).to(self.device_type)
        gn_ref = copy.deepcopy(gn)
        x_ref = torch.randn(4, 16, 8, 8, device=self.device_type, requires_grad=True)

        out_ref = gn_ref(x_ref)
        out_ref.sum().backward()

        gn.weight = nn.Parameter(
            distribute_tensor(gn.weight, device_mesh, [Replicate()])
        )
        gn.bias = nn.Parameter(
            distribute_tensor(gn.bias, device_mesh, [Replicate()])
        )
        x_dt = distribute_tensor(
            x_ref.detach().clone().requires_grad_(True), device_mesh, [Shard(0)]
        )
        out_dt = gn(x_dt)
        out_dt.sum().backward()

        w_grad = gn.weight.grad
        b_grad = gn.bias.grad
        if isinstance(w_grad, DTensor):
            w_grad = w_grad.full_tensor()
        if isinstance(b_grad, DTensor):
            b_grad = b_grad.full_tensor()
        self.assertEqual(w_grad, gn_ref.weight.grad)
        self.assertEqual(b_grad, gn_ref.bias.grad)

        x_grad = x_dt.grad
        if isinstance(x_grad, DTensor):
            x_grad = x_grad.full_tensor()
        self.assertEqual(x_grad, x_ref.grad)


if __name__ == "__main__":
    run_tests()
