# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 specialization of the generic FSDP compute-weight lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
)

from torchtitan.distributed._fsdp_weight import _FSDPWeightWithComputeRepresentation


_MXFP8_WEIGHT_TILE_SIZE = 32


@dataclass(frozen=True, slots=True)
class _MXFP8LinearOperands:
    """The independent MXFP8 tensors owned by one FSDP unshard lifetime."""

    q_weight_dgrad_NK: torch.Tensor  # noqa: N815
    s_weight_fprop_blocked: torch.Tensor
    s_weight_dgrad_blocked: torch.Tensor

    @property
    def q_weight_fprop_KN(self) -> torch.Tensor:  # noqa: N802
        return self.q_weight_dgrad_NK.t()


def _quantize_mxfp8_weight(weight_NK: torch.Tensor) -> _MXFP8LinearOperands:
    """Quantize a BF16 weight using fixed square 32x32 scale tiles."""
    if weight_NK.ndim != 2:
        raise ValueError(
            "MXFP8 32x32 weight quantization requires a 2D weight, "
            f"got {weight_NK.ndim} dimensions."
        )
    if weight_NK.dtype != torch.bfloat16:
        raise ValueError(
            "MXFP8 32x32 weight quantization requires BF16 weights, "
            f"got {weight_NK.dtype}."
        )
    if any(size % _MXFP8_WEIGHT_TILE_SIZE for size in weight_NK.shape):
        raise ValueError(
            "MXFP8 32x32 weight quantization requires both matrix dimensions "
            f"divisible by {_MXFP8_WEIGHT_TILE_SIZE}, got {tuple(weight_NK.shape)}."
        )
    (
        q_weight_dgrad_NK,
        s_weight_fprop_blocked,
        s_weight_dgrad_blocked,
    ) = triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(weight_NK.contiguous())
    return _MXFP8LinearOperands(
        q_weight_dgrad_NK=q_weight_dgrad_NK,
        s_weight_fprop_blocked=s_weight_fprop_blocked,
        s_weight_dgrad_blocked=s_weight_dgrad_blocked,
    )


class _MXFP8LinearFSDPWeight(_FSDPWeightWithComputeRepresentation):
    """BF16 FSDP parameter carrying unshard-lifetime MXFP8 operands."""

    def __init__(
        self,
        tensor: torch.Tensor,
        compute_representation: _MXFP8LinearOperands | None = None,
        **logical_metadata: Any,
    ) -> None:
        super().__init__(tensor, compute_representation, **logical_metadata)
        if compute_representation is not None:
            self._q_weight_dgrad_NK = compute_representation.q_weight_dgrad_NK
            self._s_weight_fprop_blocked = compute_representation.s_weight_fprop_blocked
            self._s_weight_dgrad_blocked = compute_representation.s_weight_dgrad_blocked

    def __tensor_flatten__(self):
        if self._tensor is not None:
            return ["_tensor"], ("sharded", self.dtype)
        return [
            "_q_weight_dgrad_NK",
            "_s_weight_fprop_blocked",
            "_s_weight_dgrad_blocked",
        ], ("compute", self.dtype)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        state, dtype = metadata
        if state == "sharded":
            return _MXFP8LinearFSDPWeight(inner_tensors["_tensor"])
        compute_representation = _MXFP8LinearOperands(
            q_weight_dgrad_NK=inner_tensors["_q_weight_dgrad_NK"],
            s_weight_fprop_blocked=inner_tensors["_s_weight_fprop_blocked"],
            s_weight_dgrad_blocked=inner_tensors["_s_weight_dgrad_blocked"],
        )
        return _MXFP8LinearFSDPWeight(
            compute_representation.q_weight_dgrad_NK,
            compute_representation,
            _logical_size=outer_size,
            _logical_stride=outer_stride,
            _logical_dtype=dtype,
            _logical_device=compute_representation.q_weight_dgrad_NK.device,
        )

    def _build_compute_representation(
        self,
        logical_weight: torch.Tensor,
        out: _MXFP8LinearOperands | None = None,
    ) -> _MXFP8LinearOperands:
        weight_NK = logical_weight
        compute_representation = _quantize_mxfp8_weight(weight_NK)
        if out is not None:
            out.q_weight_dgrad_NK.copy_(compute_representation.q_weight_dgrad_NK)
            out.s_weight_fprop_blocked.copy_(
                compute_representation.s_weight_fprop_blocked
            )
            out.s_weight_dgrad_blocked.copy_(
                compute_representation.s_weight_dgrad_blocked
            )
            return out
        return compute_representation

    def _fsdp_managed_tensors(
        self,
        compute_representation: _MXFP8LinearOperands,
    ) -> tuple[torch.Tensor, ...]:
        return (
            compute_representation.q_weight_dgrad_NK,
            compute_representation.s_weight_fprop_blocked,
            compute_representation.s_weight_dgrad_blocked,
        )

    @property
    def compute_representation(self) -> _MXFP8LinearOperands | None:
        return super().compute_representation


__all__ = [
    "_MXFP8LinearFSDPWeight",
    "_MXFP8LinearOperands",
    "_quantize_mxfp8_weight",
]
