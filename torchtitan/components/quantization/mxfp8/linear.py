# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 linear training with FSDP-managed 32x32 weight caches.

Tensor shape suffixes:
    M: flattened token rows
    N: output features
    K: input features
"""

from dataclasses import dataclass, replace
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable
from torch.distributed.tensor import DTensor

from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.sharding import LocalMapConfig

from .tensor import _MXFP8LinearFSDPWeight, _quantize_mxfp8_weight


TP = MeshAxisName.TP

_MXFP8_BLOCK_SIZE = 32
_MXFP8_SCALING_MODE = "rceil"

InputActivationFormatForBackward = Literal["bf16", "mxfp8"]
_INPUT_ACTIVATION_FORMATS_FOR_BACKWARD = ("bf16", "mxfp8")


def _pad_rows(x_MK: torch.Tensor) -> tuple[torch.Tensor, int]:
    num_rows = x_MK.shape[0]
    num_padded_rows = (
        (num_rows + _MXFP8_BLOCK_SIZE - 1) // _MXFP8_BLOCK_SIZE
    ) * _MXFP8_BLOCK_SIZE
    if num_padded_rows == num_rows:
        return x_MK, num_rows
    return F.pad(x_MK, (0, 0, 0, num_padded_rows - num_rows)), num_rows


# Adapted from torchao.prototype.moe_training.mxfp8_linear.mx_mm. This variant
# lives in TorchTitan so its autograd state and weight cache can integrate with
# FSDP and other parallelisms.
@torch._dynamo.allow_in_graph
class _MXFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        weight_NK: torch.Tensor,
        q_weight_fprop_KN: torch.Tensor,
        s_weight_fprop_blocked: torch.Tensor,
        q_weight_dgrad_NK: torch.Tensor,
        s_weight_dgrad_blocked: torch.Tensor,
        bias_N: torch.Tensor | None,
        input_activation_format_for_backward: InputActivationFormatForBackward,
    ) -> torch.Tensor:
        # FPROP always consumes rowwise MXFP8. WGRAD can either retain the
        # original BF16 input and quantize it columnwise in backward, or retain
        # a columnwise MXFP8 representation produced in forward. The former is
        # memory-safe when another operation already keeps BF16 x alive; the
        # latter reduces storage for a single-consumer input at the cost of an
        # extra cached representation when BF16 x is retained elsewhere. Under
        # full activation checkpointing, the selected state is created by
        # recompute.
        if x.dtype != torch.bfloat16 or weight_NK.dtype != torch.bfloat16:
            raise ValueError(
                "MXFP8Linear requires BF16 activations and weights; "
                f"got activation dtype {x.dtype} and weight dtype {weight_NK.dtype}."
            )
        if bias_N is not None and bias_N.dtype != torch.bfloat16:
            raise ValueError(
                f"MXFP8Linear requires a BF16 bias; got bias dtype {bias_N.dtype}."
            )
        if x.shape[-1] != weight_NK.shape[1]:
            raise ValueError(
                "MXFP8Linear activation and weight contraction dimensions must "
                f"match; got {x.shape[-1]} and {weight_NK.shape[1]}."
            )
        for name, value in (
            ("local in_features", weight_NK.shape[1]),
            ("local out_features", weight_NK.shape[0]),
        ):
            if value % _MXFP8_BLOCK_SIZE:
                raise ValueError(
                    f"MXFP8Linear requires {name} divisible by "
                    f"{_MXFP8_BLOCK_SIZE}; got {value}."
                )

        input_shape = x.shape
        x_MK, num_rows = _pad_rows(x.reshape(-1, input_shape[-1]).contiguous())
        requires_wgrad = ctx.needs_input_grad[1]
        quantize_wgrad_input_in_forward = (
            requires_wgrad and input_activation_format_for_backward == "mxfp8"
        )

        # The save format controls both computation and saved state. BF16 mode
        # requests only the rowwise FPROP operand here; backward produces the
        # columnwise WGRAD operand from the saved BF16 input.
        x_row_MK, x_col_MK, x_row_scales, x_col_scales = mxfp8_quantize_cuda(
            x_MK,
            rowwise=True,
            colwise=quantize_wgrad_input_in_forward,
            scaling_mode=_MXFP8_SCALING_MODE,
        )
        x_row_scales = triton_mx_block_rearrange(x_row_scales)
        if quantize_wgrad_input_in_forward:
            x_col_scales = triton_mx_block_rearrange(x_col_scales)

        # The 32x32 weight quantizer returns both qdata/scale pairs ready for
        # this exact BlockWise1x32 and SWIZZLE_32_4_4 B-operand contract.
        output_MN = F.scaled_mm(
            x_row_MK,
            q_weight_fprop_KN,
            scale_a=x_row_scales,
            scale_recipe_a=F.ScalingType.BlockWise1x32,
            scale_b=s_weight_fprop_blocked,
            scale_recipe_b=F.ScalingType.BlockWise1x32,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            bias=bias_N,
            output_dtype=torch.bfloat16,
        )

        # Save exactly one input-activation representation for WGRAD. BF16 mode
        # keeps the original tensor and builds the columnwise operand in
        # backward. MXFP8 mode keeps the columnwise qdata and scales produced
        # above. FPROP and DGRAD share the same weight qdata allocation.
        has_compute_weight = isinstance(weight_NK, _MXFP8LinearFSDPWeight)
        saved_weight_tensors = (
            (weight_NK,)
            if has_compute_weight
            else (q_weight_dgrad_NK, s_weight_dgrad_blocked)
        )
        if requires_wgrad and input_activation_format_for_backward == "bf16":
            ctx.save_for_backward(x, *saved_weight_tensors)
        else:
            ctx.save_for_backward(x_col_MK, x_col_scales, *saved_weight_tensors)
        ctx.input_shape = input_shape
        ctx.num_rows = num_rows
        ctx.requires_dgrad = ctx.needs_input_grad[0]
        ctx.requires_wgrad = requires_wgrad
        ctx.input_activation_format_for_backward = input_activation_format_for_backward
        ctx.has_bias = bias_N is not None
        ctx.has_compute_weight = has_compute_weight

        return output_MN[:num_rows].reshape(*input_shape[:-1], weight_NK.shape[0])

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        # WGRAD consumes either the saved columnwise activation pair or a pair
        # rebuilt from the saved BF16 input. DGRAD consumes the weight pair.
        x_hp = None
        x_col_MK = None
        x_col_scales = None
        saved_tensors = ctx.saved_tensors
        if ctx.requires_wgrad and ctx.input_activation_format_for_backward == "bf16":
            x_hp = saved_tensors[0]
            saved_weight_tensors = saved_tensors[1:]
        else:
            x_col_MK, x_col_scales = saved_tensors[:2]
            saved_weight_tensors = saved_tensors[2:]

        if ctx.has_compute_weight:
            (weight_NK,) = saved_weight_tensors
            if not isinstance(weight_NK, _MXFP8LinearFSDPWeight):
                raise RuntimeError("FSDP restored an incompatible MXFP8 weight")
            operands = weight_NK.compute_representation
            if operands is None:
                raise RuntimeError("FSDP did not build MXFP8 weight state for backward")
            q_weight_dgrad_NK = operands.q_weight_dgrad_NK
            s_weight_dgrad_blocked = operands.s_weight_dgrad_blocked
        else:
            q_weight_dgrad_NK, s_weight_dgrad_blocked = saved_weight_tensors

        grad_output_MN = grad_output.contiguous().reshape(-1, grad_output.shape[-1])
        grad_bias_N = grad_output_MN.sum(dim=0) if ctx.has_bias else None

        grad_input = None
        grad_weight_NK = None
        if ctx.requires_dgrad or ctx.requires_wgrad:
            padded_grad_output_MN, _ = _pad_rows(grad_output_MN)
            (
                grad_output_row_MN,
                grad_output_col_MN,
                grad_output_row_scales,
                grad_output_col_scales,
            ) = mxfp8_quantize_cuda(
                padded_grad_output_MN,
                rowwise=ctx.requires_dgrad,
                colwise=ctx.requires_wgrad,
                scaling_mode=_MXFP8_SCALING_MODE,
            )

            if ctx.requires_dgrad:
                grad_output_row_scales = triton_mx_block_rearrange(
                    grad_output_row_scales
                )
                grad_input_MK = F.scaled_mm(
                    grad_output_row_MN,
                    q_weight_dgrad_NK,
                    scale_a=grad_output_row_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=s_weight_dgrad_blocked,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )
                grad_input = grad_input_MK[: ctx.num_rows].reshape(ctx.input_shape)

            if ctx.requires_wgrad:
                if ctx.input_activation_format_for_backward == "bf16":
                    assert x_hp is not None
                    x_MK, _ = _pad_rows(
                        x_hp.reshape(-1, ctx.input_shape[-1]).contiguous()
                    )
                    _, x_col_MK, _, x_col_scales = mxfp8_quantize_cuda(
                        x_MK,
                        rowwise=False,
                        colwise=True,
                        scaling_mode=_MXFP8_SCALING_MODE,
                    )
                    x_col_scales = triton_mx_block_rearrange(x_col_scales)

                assert x_col_MK is not None
                assert x_col_scales is not None
                grad_output_col_scales = triton_mx_block_rearrange(
                    grad_output_col_scales
                )
                grad_weight_NK = F.scaled_mm(
                    grad_output_col_MN.t(),
                    x_col_MK,
                    scale_a=grad_output_col_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=x_col_scales,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )

        return grad_input, grad_weight_NK, None, None, None, None, grad_bias_N, None


spmd.register_local_autograd_function(_MXFP8LinearFunction)


class MXFP8Linear(Linear):
    """Linear using 1D activations and cached 32x32 weight quantization."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for ``Linear.Config``."""

        input_activation_format_for_backward: InputActivationFormatForBackward = "bf16"
        """Format used to save the input activation needed by WGRAD.

        ``"bf16"`` saves the original input and quantizes it columnwise during
        backward. ``"mxfp8"`` produces the columnwise representation during
        forward and saves its qdata and scales for backward.
        """

        def __post_init__(self) -> None:
            if (
                self.input_activation_format_for_backward
                not in _INPUT_ACTIVATION_FORMATS_FOR_BACKWARD
            ):
                raise ValueError(
                    "MXFP8 input_activation_format_for_backward must be one of "
                    f"{_INPUT_ACTIVATION_FORMATS_FOR_BACKWARD}; got "
                    f"{self.input_activation_format_for_backward!r}."
                )
            for name in ("in_features", "out_features"):
                value = getattr(self, name)
                if value % _MXFP8_BLOCK_SIZE:
                    raise ValueError(
                        f"MXFP8 requires {name} divisible by {_MXFP8_BLOCK_SIZE}; "
                        f"got {name}={value}."
                    )

        def build(self, **kwargs):
            # Model converters run before update_from_config() attaches the
            # stock Linear sharding config. Adapt that late-bound config here so
            # the opaque MXFP8 autograd function runs on local tensors with the
            # correct TP input and input-gradient placements.
            instance = Linear.Config.build(self, **kwargs)
            if instance._sharding_config is not None:
                sharding_config = instance._sharding_config
                weight_tp = (
                    sharding_config.state_shardings["weight"]
                    .per_axis_spmd_types()
                    .get(TP)
                )
                rowwise = isinstance(weight_tp, spmd.Shard) and weight_tp.dim == 1
                if rowwise:
                    input_layout = dense_activation_placement(
                        tp=spmd.S(-1), cp=spmd.S(1)
                    )
                    input_grad_layout = dense_activation_placement(
                        tp=spmd.S(-1), cp=spmd.S(1)
                    )
                else:
                    input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(1))
                    input_grad_layout = dense_activation_placement(
                        tp=spmd.P, cp=spmd.S(1)
                    )
                instance._sharding_config = replace(
                    sharding_config,
                    in_src_shardings={
                        **(sharding_config.in_src_shardings or {}),
                        "input": input_layout,
                    },
                    in_dst_shardings={
                        **(sharding_config.in_dst_shardings or {}),
                        "input": input_layout,
                    },
                    local_map=LocalMapConfig(in_grad_placements=(input_grad_layout,)),
                )
            return instance

    def __init__(self, config: Config):
        super().__init__(config)
        self.input_activation_format_for_backward = (
            config.input_activation_format_for_backward
        )

    def configure_fsdp(self) -> None:
        """Tie MXFP8 operands to each FSDP unshard lifetime."""
        weight = self.weight
        if isinstance(weight, DTensor):
            local_weight = weight._local_tensor
            if isinstance(local_weight, _MXFP8LinearFSDPWeight):
                return
            wrapped_weight = DTensor.from_local(
                _MXFP8LinearFSDPWeight(local_weight),
                weight.device_mesh,
                weight.placements,
                run_check=False,
                shape=weight.shape,
                stride=weight.stride(),
            )
        else:
            if isinstance(weight, _MXFP8LinearFSDPWeight):
                return
            wrapped_weight = _MXFP8LinearFSDPWeight(weight)
        torch.utils.swap_tensors(
            self.weight,
            nn.Parameter(wrapped_weight, requires_grad=self.weight.requires_grad),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight_NK = weight._local_tensor if isinstance(weight, DTensor) else weight
        if isinstance(weight_NK, _MXFP8LinearFSDPWeight):
            operands = weight_NK.compute_representation
            if operands is None:
                raise RuntimeError("MXFP8 FSDP weight has no compute representation")
        else:
            # Without FSDP, the weight remains an ordinary BF16 parameter and
            # its MXFP8 operands are built for each linear invocation.
            with torch.no_grad():
                operands = _quantize_mxfp8_weight(weight_NK)
        return _MXFP8LinearFunction.apply(
            input,
            weight_NK,
            operands.q_weight_fprop_KN,
            operands.s_weight_fprop_blocked,
            operands.q_weight_dgrad_NK,
            operands.s_weight_dgrad_blocked,
            self.bias,
            self.input_activation_format_for_backward,
        )


__all__ = ["InputActivationFormatForBackward", "MXFP8Linear"]
