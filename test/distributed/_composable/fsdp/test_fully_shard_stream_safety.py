# Owner(s): ["oncall: distributed"]
"""FSDP2 stream-ordering validation via the CUDA sanitizer.

Uses the sanitizer's vector clock model as an oracle to verify that FSDP2's
~40 cross-stream sync edges (across 4 dedicated CUDA streams) are correct.
Covers 13 critical bridges across 7 data flows identified by formal analysis
of the sync DAG (see plan for details).

If the sanitizer fires: real bug in FSDP2 stream ordering.
If all configs pass: first systematic stream-ordering regression test.
"""

import functools

import torch
import torch.cuda._sanitizer as csan
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests


class TestFSDP2StreamSafety(FSDPTest):
    """Validates FSDP2 stream ordering using the CUDA sanitizer as oracle.

    The sanitizer tracks wait_stream/wait_event/record_event via vector clocks
    and raises CUDASanitizerErrors if an unsynchronized cross-stream access is
    detected.  Each test runs a full training loop and asserts 0 errors.
    """

    @property
    def world_size(self) -> int:
        return 2

    def _run_fsdp_with_sanitizer(
        self,
        model: nn.Module,
        inp_fn,
        n_steps: int = 4,
        optim_cls=torch.optim.Adam,
        backward_fn=None,
    ) -> int:
        """Run training steps under the CUDA sanitizer and return error count.

        Uses the sanitizer's accumulate mode so all races across all steps
        are collected without raising.
        """
        with csan.cuda_sanitizer as san:
            optim = optim_cls(model.parameters(), lr=1e-3)
            for step in range(n_steps):
                if backward_fn is not None:
                    backward_fn(model, optim, step)
                else:
                    optim.zero_grad()
                    loss = model(inp_fn()).sum()
                    loss.backward()
                    optim.step()
                torch.cuda.synchronize()
        return len(san.errors)

    # ------------------------------------------------------------------
    # Phase 1: Core paths
    # ------------------------------------------------------------------

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_fsdp_basic(self):
        """Test 1: 1D FSDP, eager, 4 training steps.

        Bridges exercised: F1b/d (cross-iter D->CI/AG), F2 (CI->AG),
        G1 (D->RS), H1 (RS->D finalize), P1/P2 (prefetch reuse), R2 (drain).
        """
        torch.manual_seed(42)
        dim = 16
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module)
        fully_shard(model)
        model = model.cuda()

        errors = self._run_fsdp_with_sanitizer(
            model,
            inp_fn=lambda: torch.randn(4, dim, device="cuda"),
            n_steps=4,
        )
        self.assertEqual(errors, 0, f"Sanitizer detected {errors} stream race(s)")

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_hsdp(self):
        """Test 2: HSDP with mesh (1, 2).

        Bridges exercised: F1b/d, F2, G1, H2 (RS->AR), H3 (AR->D finalize),
        P1/P2, R2.  With world_size=2, replicate_size=1 makes AR a no-op but
        the stream edges still fire (code at _fsdp_collectives.py:653-654 is
        unconditional).
        """
        torch.manual_seed(42)
        dim = 16
        mesh = init_device_mesh(
            "cuda",
            (1, self.world_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)
        model = model.cuda()

        errors = self._run_fsdp_with_sanitizer(
            model,
            inp_fn=lambda: torch.randn(4, dim, device="cuda"),
            n_steps=4,
        )
        self.assertEqual(errors, 0, f"Sanitizer detected {errors} stream race(s)")

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_grad_accumulation(self):
        """Test 3: 1D FSDP with 2 micro-batches (gradient accumulation).

        Bridges exercised: A1 (_wait_for_post_backward cross-micro-batch),
        plus F1b/d, F2, G1, H1, P1/P2, R2.

        This is the highest bug-probability test -- PR #183983 acknowledged
        no good way to test the _last_post_reduce_events optimization.
        """
        torch.manual_seed(42)
        dim = 16
        n_microbatches = 2
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module)
        fully_shard(model)
        model = model.cuda()

        def grad_accum_backward(model, optim, step):
            optim.zero_grad()
            for micro in range(n_microbatches):
                is_last = micro == n_microbatches - 1
                model.set_requires_gradient_sync(is_last)
                model.set_is_last_backward(is_last)
                inp = torch.randn(4, dim, device="cuda")
                loss = model(inp).sum()
                loss.backward()
            optim.step()

        errors = self._run_fsdp_with_sanitizer(
            model,
            inp_fn=None,
            n_steps=4,
            backward_fn=grad_accum_backward,
        )
        self.assertEqual(errors, 0, f"Sanitizer detected {errors} stream race(s)")

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_rs_max_input_buffers(self):
        """Test 4: 1D FSDP with reduce_scatter_max_input_buffers=2.

        Bridges exercised: R1 (deferred RS buffer reclamation -- sole test),
        plus F1b/d, F2, G1, H1, P1/P2, R2.

        With max_input_buffers=2 and 3 layers, the RS buffer reclamation code
        path at _fsdp_param_group.py:666-680 is exercised: layer 2's RS buffer
        must be waited-on before layer 0's RS can reuse it.
        """
        torch.manual_seed(42)
        dim = 16
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module)
        fully_shard(model)
        model.set_reduce_scatter_max_input_buffers(2)
        model = model.cuda()

        errors = self._run_fsdp_with_sanitizer(
            model,
            inp_fn=lambda: torch.randn(4, dim, device="cuda"),
            n_steps=4,
        )
        self.assertEqual(errors, 0, f"Sanitizer detected {errors} stream race(s)")

    # ------------------------------------------------------------------
    # Phase 2: Interaction paths
    # ------------------------------------------------------------------

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_mixed_precision(self):
        """Test 5: Mixed precision (param_dtype=bf16, reduce_dtype=fp32).

        Bridges exercised: F1b/d, F2, G1, H1, P1/P2, R2, plus dtype cast
        ordering between streams.
        """
        torch.manual_seed(42)
        dim = 16
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)
        model = model.cuda()

        errors = self._run_fsdp_with_sanitizer(
            model,
            inp_fn=lambda: torch.randn(4, dim, device="cuda"),
            n_steps=4,
        )
        self.assertEqual(errors, 0, f"Sanitizer detected {errors} stream race(s)")

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_reshard_after_forward(self):
        """Test 6: reshard_after_forward=True.

        Bridges exercised: F1b/d, F2, G1, H1, P1/P2, R2 + reshard event path.
        """
        torch.manual_seed(42)
        dim = 16
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=True)
        model = model.cuda()

        errors = self._run_fsdp_with_sanitizer(
            model,
            inp_fn=lambda: torch.randn(4, dim, device="cuda"),
            n_steps=4,
        )
        self.assertEqual(errors, 0, f"Sanitizer detected {errors} stream race(s)")

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_separate_rs_group(self):
        """Test 7: set_separate_reduce_scatter_group().

        Bridges exercised: F1b/d, F2, G1, H1, P1/P2, R2.  Same edges but
        reduce-scatter uses a separate NCCL communicator.
        """
        torch.manual_seed(42)
        dim = 16
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module)
        fully_shard(model)
        model.set_separate_reduce_scatter_group()
        model = model.cuda()

        errors = self._run_fsdp_with_sanitizer(
            model,
            inp_fn=lambda: torch.randn(4, dim, device="cuda"),
            n_steps=4,
        )
        self.assertEqual(errors, 0, f"Sanitizer detected {errors} stream race(s)")

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_hsdp_grad_accumulation(self):
        """Test 8: HSDP + gradient accumulation with set_requires_all_reduce.

        Bridges exercised: A1, H2, H3 + early-return path at
        _fsdp_collectives.py:636-649 when all_reduce_grads=False.
        """
        torch.manual_seed(42)
        dim = 16
        n_microbatches = 2
        mesh = init_device_mesh(
            "cuda",
            (1, self.world_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)
        model = model.cuda()

        def hsdp_grad_accum_backward(model, optim, step):
            optim.zero_grad()
            for micro in range(n_microbatches):
                is_last = micro == n_microbatches - 1
                model.set_requires_all_reduce(is_last)
                model.set_is_last_backward(is_last)
                inp = torch.randn(4, dim, device="cuda")
                loss = model(inp).sum()
                loss.backward()
            optim.step()

        errors = self._run_fsdp_with_sanitizer(
            model,
            inp_fn=None,
            n_steps=4,
            backward_fn=hsdp_grad_accum_backward,
        )
        self.assertEqual(errors, 0, f"Sanitizer detected {errors} stream race(s)")

    # ------------------------------------------------------------------
    # Phase 3: Edge cases
    # ------------------------------------------------------------------

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_post_optim_event(self):
        """Test 9: set_post_optim_event() between optim.step() and forward.

        Bridges exercised: F1a/c (post_optim_event variant -- sole test).
        """
        torch.manual_seed(42)
        dim = 16
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        for module in model:
            fully_shard(module)
        fully_shard(model)
        model = model.cuda()

        def step_post_hook(fsdp_module, opt, args, kwargs):
            post_optim_event = torch.cuda.current_stream().record_event()
            fsdp_module.set_post_optim_event(post_optim_event)

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        optim.register_step_post_hook(functools.partial(step_post_hook, model))

        with csan.cuda_sanitizer as san:
            for _ in range(4):
                optim.zero_grad()
                inp = torch.randn(4, dim, device="cuda")
                loss = model(inp).sum()
                loss.backward()
                optim.step()
                torch.cuda.synchronize()
        self.assertEqual(
            len(san.errors), 0, f"Sanitizer detected {len(san.errors)} stream race(s)"
        )

    @skip_if_lt_x_gpu(2)
    def test_stream_safety_partial_group_backward(self):
        """Test 10: Chunked loss (partial-group backward).

        Bridges exercised: Q1 (partial-group _post_reduce_event wait -- sole
        test).  Between partial-group post_backwards, autograd runs on D.
        Without Q1, the caching allocator could hand autograd a block that RS
        is still writing.
        """
        torch.manual_seed(42)
        dim, vocab_size, n_chunks = 32, 128, 2

        model = _ChunkedHeadModel(dim, vocab_size).cuda()
        with torch.no_grad():
            for p in model.parameters():
                dist.broadcast(p, src=0)
        fully_shard(model.embed)
        fully_shard(model.body)
        fully_shard(model.head)
        fully_shard(model)

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        with csan.cuda_sanitizer as san:
            for _ in range(4):
                optim.zero_grad()
                tokens = torch.randint(0, vocab_size, (2, 16), device="cuda")
                h = model(tokens, skip_head=True)
                h_grads = []
                for chunk in torch.chunk(h.detach(), n_chunks, dim=1):
                    chunk = chunk.contiguous().detach().requires_grad_(True)
                    model.head(chunk).sum().backward()
                    h_grads.append(chunk.grad.detach())
                h.backward(torch.cat(h_grads, dim=1))
                optim.step()
                torch.cuda.synchronize()
        self.assertEqual(
            len(san.errors), 0, f"Sanitizer detected {len(san.errors)} stream race(s)"
        )


class _ChunkedHeadModel(nn.Module):
    """Minimal model for chunked-loss (partial-group backward) testing."""

    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.body = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens, *, skip_head=False):
        h = self.embed(tokens)
        h = self.body(h)
        if skip_head:
            return h
        return self.head(h)


if __name__ == "__main__":
    run_tests()
