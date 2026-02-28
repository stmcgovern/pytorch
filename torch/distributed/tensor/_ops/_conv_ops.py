# Copyright (c) Meta Platforms, Inc. and affiliates
# implement convolution and batch norm ops for distributed tensor

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule


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
    input_spec = op_schema.args_schema[0]
    (
        grad_output_spec,
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

    assert isinstance(grad_output_spec, DTensorSpec)
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

    # TODO: actually the output_mask is not respected here, we should
    # set the corresponding spec to `None` if the output_mask is not `False`
    # for a certain output Tensor. This also applies to the conv handler
    # in torch/distributed/tensor/_tp_conv.py
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])


def _batch_norm_fwd_rules(op_schema: OpSchema) -> OutputSharding:
    """Sharding propagation for native_batch_norm forward variants.

    Supports batch-dim sharding (each rank normalizes its local batch — same
    semantics as non-synchronized batch norm in data parallelism) and
    channel-dim sharding (each rank processes independent channels).

    Args schema: input, weight?, bias?, running_mean?, running_var?,
                 training, momentum, eps
    Returns: (output, save_mean, save_invstd)
    """
    input_spec = op_schema.args_schema[0]
    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    in_shape = input_spec.tensor_meta.shape
    output_stride = _compute_contiguous_stride(in_shape)

    # output has same shape and sharding as input
    output_meta = TensorMeta(
        torch.Size(in_shape), output_stride, input_spec.tensor_meta.dtype
    )
    output_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        input_spec.dim_map,
        input_spec.sums,
        tensor_meta=output_meta,
    )

    # save_mean/save_invstd have shape (C,) — follow the channel dim sharding
    n_channels = in_shape[1] if len(in_shape) > 1 else 0
    channel_mesh_dim = input_spec.dim_map[1] if len(input_spec.dim_map) > 1 else -1
    stats_meta = TensorMeta(
        torch.Size([n_channels]), (1,), input_spec.tensor_meta.dtype
    )
    stats_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [channel_mesh_dim],
        [],
        tensor_meta=stats_meta,
    )

    return OutputSharding([output_spec, stats_spec, stats_spec])


@register_prop_rule(aten._batch_norm_with_update.default)
def batch_norm_with_update_rules(op_schema: OpSchema) -> OutputSharding:
    """Sharding propagation for _batch_norm_with_update.

    Args schema: input, weight?, bias?, running_mean, running_var,
                 momentum, eps
    Returns: (output, save_mean, save_rstd, reserve)
    """
    fwd = _batch_norm_fwd_rules(op_schema)
    assert isinstance(fwd.output_spec, list)
    output_spec, stats_spec, _ = fwd.output_spec

    # reserve is an empty (0,) uint8 tensor — always replicated
    reserve_meta = TensorMeta(torch.Size([0]), (1,), torch.uint8)
    reserve_spec = DTensorSpec.from_dim_map(
        op_schema.args_schema[0].mesh,  # type: ignore[union-attr]
        [-1],
        [],
        tensor_meta=reserve_meta,
    )

    return OutputSharding([output_spec, stats_spec, stats_spec, reserve_spec])


@register_prop_rule(aten.native_batch_norm.default)
def native_batch_norm_rules(op_schema: OpSchema) -> OutputSharding:
    return _batch_norm_fwd_rules(op_schema)


@register_prop_rule(aten._native_batch_norm_legit.default)
def native_batch_norm_legit_rules(op_schema: OpSchema) -> OutputSharding:
    return _batch_norm_fwd_rules(op_schema)


@register_prop_rule(aten._native_batch_norm_legit.no_stats)
def native_batch_norm_legit_no_stats_rules(op_schema: OpSchema) -> OutputSharding:
    return _batch_norm_fwd_rules(op_schema)


@register_prop_rule(aten.native_batch_norm_backward.default)
def native_batch_norm_backward_rules(op_schema: OpSchema) -> OutputSharding:
    """Sharding propagation for native_batch_norm_backward.

    Args schema: grad_out, input, weight?, running_mean?, running_var?,
                 save_mean?, save_invstd?, train, eps, output_mask
    Returns: (grad_input, grad_weight, grad_bias)
    """
    (
        grad_out_spec,
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

    assert isinstance(grad_out_spec, DTensorSpec)
    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    # grad_input: same shape/sharding as input
    grad_input_spec = input_spec

    # grad_weight/grad_bias: shape (C,), need reduction across sharded dims
    n_channels = input_spec.tensor_meta.shape[1]
    param_meta = TensorMeta(
        torch.Size([n_channels]), (1,), input_spec.tensor_meta.dtype
    )

    if weight_spec is not None:
        grad_weight_spec = DTensorSpec.from_dim_map(
            input_spec.mesh,
            [-1],
            [0],
            tensor_meta=param_meta,
        )
    else:
        grad_weight_spec = None

    # grad_bias always produced (even if bias was None, PyTorch still returns it)
    grad_bias_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1],
        [0],
        tensor_meta=param_meta,
    )

    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])
