# Copyright (c) Meta Platforms, Inc. and affiliates
# implement convolution and batch norm ops for distributed tensor

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, OpSchema, OutputSharding
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    batch_dim_strategies,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import register_prop_rule
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Placement, Replicate


aten = torch.ops.aten


def _compute_contiguous_stride(shape: list[int] | torch.Size) -> tuple[int, ...]:
    stride = [1]
    for i in range(1, len(shape)):
        stride.insert(0, stride[0] * shape[-i])
    return tuple(stride)


@register_prop_rule(aten.convolution.default)
def convolution_rules(op_schema: OpSchema) -> OutputSharding:
    (
        input_spec,
        weight_spec,
        bias_spec,
        stride,
        padding,
        dilation,
        _transposed,
        _output_padding,
        _groups,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    # bias_spec can be None (optional parameter in aten.convolution schema)
    if bias_spec is not None:
        assert isinstance(bias_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None
    assert weight_spec.tensor_meta is not None
    in_shape = input_spec.tensor_meta.shape
    weight_shape = weight_spec.tensor_meta.shape
    assert isinstance(stride, list), f"stride must be list, got {type(stride)}"
    assert isinstance(padding, list), f"padding must be list, got {type(padding)}"
    assert isinstance(dilation, list), f"dilation must be list, got {type(dilation)}"
    # weight_shape might not be torch.Size in all cases (e.g., SymIntArrayRef during tracing)
    # so we don't assert its type, just use it
    out_conv_shape = [
        (d + 2 * padding[i] - dilation[i] * (weight_shape[i + 1] - 1) - 1) // stride[i]
        + 1
        for (i, d) in enumerate(in_shape[2:])
    ]
    output_shape = [in_shape[0], weight_shape[0]] + out_conv_shape
    output_stride = _compute_contiguous_stride(output_shape)
    output_dim_map = input_spec.dim_map
    pending_sums = input_spec.sums

    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        tuple(output_stride),
        input_spec.tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            output_dim_map,
            pending_sums,
            tensor_meta=tensor_meta,
        )
    )


@register_prop_rule(aten.convolution_backward.default)
def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding:
    (
        _grad_output_spec,
        input_spec,
        weight_spec,
        bias_shape_opt,
        _stride,
        _padding,
        _dilation,
        _transposed,
        _output_padding,
        _groups,
        _output_mask,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    # bias_shape_opt can be None (optional parameter in aten.convolution_backward schema)
    if bias_shape_opt is not None:
        assert isinstance(bias_shape_opt, list)
    assert input_spec.tensor_meta is not None
    weight_tensor_meta = weight_spec.tensor_meta

    # Only create bias_tensor_meta if bias_shape_opt is not None
    if bias_shape_opt is not None:
        bias_tensor_meta = TensorMeta(
            torch.Size(bias_shape_opt),
            (1,),
            input_spec.tensor_meta.dtype,
        )
    else:
        bias_tensor_meta = None

    grad_input_spec = input_spec
    grad_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1, -1, -1, -1],
        [0],
        tensor_meta=weight_tensor_meta,
    )

    # Only create grad_bias_spec if we have bias_tensor_meta
    if bias_tensor_meta is not None:
        grad_bias_spec = DTensorSpec.from_dim_map(
            input_spec.mesh,
            [-1],
            [0],
            tensor_meta=bias_tensor_meta,
        )
    else:
        grad_bias_spec = None

    # TODO: output_mask is not respected here — we should set the
    # corresponding spec to None when output_mask is False.
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])


# --------------------------------------------------------------------------- #
# Batch norm: single-dim strategies
# --------------------------------------------------------------------------- #


def _get_bn_training(op: OpOverload, args_schema: ArgsType) -> bool:
    """Extract training mode from batch norm args."""
    if op == aten._batch_norm_with_update.default:
        return True
    if op == aten._batch_norm_no_update.default:
        return False
    for i, schema_arg in enumerate(op._schema.arguments):
        if schema_arg.name in ("training", "train"):
            return args_schema[i]  # type: ignore[return-value]
    return True


def _bn_fwd_strategies(
    training: bool,
    ndim: int,
    num_outputs: int,
    num_tensor_inputs: int,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Generate batch norm forward strategies for a single mesh dim.

    Batch norm tensors fall into two categories:
      - ndim-D: input and output (share the same sharding)
      - 1D: weight, bias, running_mean, running_var, save_mean, save_invstd

    Channel-dim sharding (dim 1 on ndim-D, dim 0 on 1D) is always valid since
    per-channel statistics are independent. In inference mode, BN is a per-channel
    pointwise op so any dim works. In training mode, batch and spatial dim sharding
    compute local statistics (DDP semantics for batch dim; wrong for spatial dims).
    """
    num_param_inputs = num_tensor_inputs - 1
    has_reserve = num_outputs == 4
    # stats outputs: save_mean, save_invstd (exclude reserve if present)
    num_stats_outputs = num_outputs - 1 - int(has_reserve)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    # Channel-dim sharding: always valid
    ch: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(1)]
    ch.extend([_ShardingPlaceholder(0)] * num_stats_outputs)
    if has_reserve:
        ch.append(Replicate())
    ch.append(_ShardingPlaceholder(1))
    ch.extend([_ShardingPlaceholder(0)] * num_param_inputs)
    strategies.append(ch)

    if not training:
        # Inference: BN is pointwise with running stats, all dims valid
        for d in range(ndim):
            if d == 1:
                continue
            s: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(d)]
            s.extend([Replicate()] * num_stats_outputs)
            if has_reserve:
                s.append(Replicate())
            s.append(_ShardingPlaceholder(d))
            s.extend([Replicate()] * num_param_inputs)
            strategies.append(s)
    else:
        # Training: batch-dim sharding (DDP semantics — local statistics)
        b: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(0)]
        b.extend([Replicate()] * num_stats_outputs)
        if has_reserve:
            b.append(Replicate())
        b.append(_ShardingPlaceholder(0))
        b.extend([Replicate()] * num_param_inputs)
        strategies.append(b)

    return strategies


@register_single_dim_strategy(
    [
        aten.native_batch_norm.default,
        aten._native_batch_norm_legit.default,
        aten._native_batch_norm_legit.no_stats,
        aten._batch_norm_with_update.default,
        aten._batch_norm_no_update.default,
    ],
)
def batch_norm_fwd_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    assert isinstance(input_meta, TensorMeta)
    ndim = len(input_meta.shape)
    training = _get_bn_training(op, args_schema)
    num_outputs = len(op._schema.returns)
    num_tensor_inputs = sum(1 for a in args_schema if isinstance(a, TensorMeta))
    return _bn_fwd_strategies(training, ndim, num_outputs, num_tensor_inputs)


@register_prop_rule(aten.native_batch_norm_backward.default)
def native_batch_norm_backward_rules(op_schema: OpSchema) -> OutputSharding:
    """Returns: (grad_input, grad_weight, grad_bias)"""
    (
        _grad_out_spec,
        input_spec,
        weight_spec,
        _running_mean_spec,
        _running_var_spec,
        _save_mean_spec,
        _save_invstd_spec,
        _train,
        _eps,
        _output_mask,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    grad_input_spec = input_spec

    # grad_weight/grad_bias: shape (C,), partial sum across batch/spatial dims
    n_channels = input_spec.tensor_meta.shape[1]
    param_meta = TensorMeta(
        torch.Size([n_channels]), (1,), input_spec.tensor_meta.dtype
    )

    if weight_spec is not None:
        grad_weight_spec = DTensorSpec.from_dim_map(
            input_spec.mesh, [-1], [0], tensor_meta=param_meta
        )
    else:
        grad_weight_spec = None

    grad_bias_spec = DTensorSpec.from_dim_map(
        input_spec.mesh, [-1], [0], tensor_meta=param_meta
    )

    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])


# --------------------------------------------------------------------------- #
# Group norm: per-sample, per-group stats → only batch-dim sharding is valid.
# Unlike batch norm, group norm takes explicit N/C/HxW scalar args that must
# be adjusted to the local batch size when the batch dim is sharded.
# --------------------------------------------------------------------------- #


@register_prop_rule(aten.native_group_norm.default, skip_decomp=True)
def native_group_norm_rules(op_schema: OpSchema) -> OutputSharding:
    """Forward: native_group_norm(input, weight?, bias?, N, C, HxW, group, eps)
    -> (output, mean, rstd)"""
    (
        input_spec,
        weight_spec,
        bias_spec,
        N,
        _C,
        _HxW,
        group,
        _eps,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    # Build redistribution suggestions: input allows only batch-dim sharding,
    # weight/bias must be Replicate.
    suggest_args = list(op_schema.args_schema)
    need_redistribute = False

    input_unsupported = any(d != -1 for d in input_spec.dim_map[1:]) or bool(
        input_spec.sums
    )
    if input_unsupported:
        suggest_args[0] = DTensorSpec(
            input_spec.mesh,
            tuple(Replicate() for _ in input_spec.placements),
            input_spec.tensor_meta,
        )
        need_redistribute = True

    for idx, spec in ((1, weight_spec), (2, bias_spec)):
        if isinstance(spec, DTensorSpec) and not all(
            isinstance(p, Replicate) for p in spec.placements
        ):
            suggest_args[idx] = DTensorSpec(
                spec.mesh,
                tuple(Replicate() for _ in spec.placements),
                spec.tensor_meta,
            )
            need_redistribute = True

    # Compute output specs based on the TARGET input placement
    target_input = suggest_args[0] if input_unsupported else input_spec
    assert isinstance(target_input, DTensorSpec)

    output_spec = DTensorSpec(
        target_input.mesh, target_input.placements, target_input.tensor_meta
    )

    stats_meta = TensorMeta(
        torch.Size([N, group]),
        (group, 1),
        input_spec.tensor_meta.dtype,
    )
    stats_spec = DTensorSpec.from_dim_map(
        target_input.mesh,
        [target_input.dim_map[0], -1],
        [],
        tensor_meta=stats_meta,
    )
    output_specs = [output_spec, stats_spec, stats_spec]

    # Adjust scalar N to local batch size when batch-dim is sharded
    if target_input.dim_map[0] != -1:
        local_shape, _ = compute_local_shape_and_global_offset(
            target_input.tensor_meta.shape,
            target_input.mesh,
            target_input.placements,
            skip_offset=True,
        )
        suggest_args[3] = local_shape[0]
        need_redistribute = True

    if need_redistribute:
        return OutputSharding(
            output_specs,
            redistribute_schema=OpSchema(
                op_schema.op, tuple(suggest_args), op_schema.kwargs_schema
            ),
            needs_redistribute=True,
            use_val_from_redistribute_schema=True,
        )

    return OutputSharding(output_specs)


@register_prop_rule(aten.native_group_norm_backward.default, skip_decomp=True)
def native_group_norm_backward_rules(op_schema: OpSchema) -> OutputSharding:
    """Backward: native_group_norm_backward(grad_out, input, mean, rstd,
    weight?, N, C, HxW, group, output_mask) -> (grad_input, grad_weight, grad_bias)"""
    (
        grad_out_spec,
        input_spec,
        _mean_spec,
        _rstd_spec,
        weight_spec,
        _N,
        _C,
        _HxW,
        _group,
        _output_mask,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    suggest_args = list(op_schema.args_schema)
    need_redistribute = False

    # Only batch-dim sharding is valid (same constraint as forward).
    # If input has non-batch sharding, redirect to Replicate.
    input_unsupported = any(d != -1 for d in input_spec.dim_map[1:]) or bool(
        input_spec.sums
    )
    if input_unsupported:
        replicate_placements = tuple(Replicate() for _ in input_spec.placements)
        suggest_args[1] = DTensorSpec(
            input_spec.mesh, replicate_placements, input_spec.tensor_meta
        )
        need_redistribute = True

    # Use the target placement for output specs (after potential redistribution)
    target_input = suggest_args[1] if input_unsupported else input_spec
    assert isinstance(target_input, DTensorSpec)

    # grad_out must match input's target placement
    if isinstance(grad_out_spec, DTensorSpec):
        if grad_out_spec.placements != target_input.placements:
            suggest_args[0] = DTensorSpec(
                grad_out_spec.mesh, target_input.placements, grad_out_spec.tensor_meta
            )
            need_redistribute = True

    # weight must be Replicate
    if isinstance(weight_spec, DTensorSpec) and not all(
        isinstance(p, Replicate) for p in weight_spec.placements
    ):
        suggest_args[4] = DTensorSpec(
            weight_spec.mesh,
            tuple(Replicate() for _ in weight_spec.placements),
            weight_spec.tensor_meta,
        )
        need_redistribute = True

    n_channels = input_spec.tensor_meta.shape[1]
    param_meta = TensorMeta(
        torch.Size([n_channels]), (1,), input_spec.tensor_meta.dtype
    )

    # grad_weight/grad_bias need Partial("sum") on the batch-sharded mesh dim
    batch_mesh_dim = target_input.dim_map[0]
    sums = [batch_mesh_dim] if batch_mesh_dim != -1 else []

    grad_input_spec = target_input

    if isinstance(weight_spec, DTensorSpec):
        grad_weight_spec = DTensorSpec.from_dim_map(
            input_spec.mesh, [-1], sums, tensor_meta=param_meta
        )
    else:
        grad_weight_spec = None

    grad_bias_spec = DTensorSpec.from_dim_map(
        input_spec.mesh, [-1], sums, tensor_meta=param_meta
    )

    output_specs = [grad_input_spec, grad_weight_spec, grad_bias_spec]

    # Adjust scalar N to local batch size when batch-dim is sharded
    if batch_mesh_dim != -1:
        local_shape, _ = compute_local_shape_and_global_offset(
            target_input.tensor_meta.shape,
            target_input.mesh,
            target_input.placements,
            skip_offset=True,
        )
        suggest_args[5] = local_shape[0]  # local N (index 5 in backward)
        need_redistribute = True

    if need_redistribute:
        return OutputSharding(
            output_specs,
            redistribute_schema=OpSchema(
                op_schema.op, tuple(suggest_args), op_schema.kwargs_schema
            ),
            needs_redistribute=True,
            use_val_from_redistribute_schema=True,
        )

    return OutputSharding(output_specs)


# --------------------------------------------------------------------------- #
# MaxPool2d: last 2 dims (spatial) coupled by pooling kernel;
# all leading dims independently shardable.
# --------------------------------------------------------------------------- #


@register_single_dim_strategy([aten.max_pool2d_with_indices.default])
def max_pool2d_fwd_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    assert isinstance(input_meta, TensorMeta)
    # 2 outputs (values, indices) + 1 input; last 2 dims are spatial
    return batch_dim_strategies(len(input_meta.shape) - 2, num_slots=3)


@register_single_dim_strategy([aten.max_pool2d_with_indices_backward.default])
def max_pool2d_bwd_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[1]  # self (the forward input)
    assert isinstance(input_meta, TensorMeta)
    # 1 output + 3 tensor inputs (grad_output, self, indices); last 2 dims are spatial
    return batch_dim_strategies(len(input_meta.shape) - 2, num_slots=4)
