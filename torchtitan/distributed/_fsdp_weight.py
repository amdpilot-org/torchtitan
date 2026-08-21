# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private FSDP lifecycle support for compute weight representations."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


_FSDP_SUBCLASS_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.c10d.scatter_.default,
}

_FSDP_COMPUTE_VIEW_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
}

_FSDP_COMPUTE_FACTORY_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.zeros_like.default,
}


def _validate_refilled_tensor_identity(
    managed_tensors: tuple[torch.Tensor, ...],
    refilled_tensors: tuple[torch.Tensor, ...],
) -> None:
    """Require a refill to preserve every tensor object managed by FSDP."""
    if len(managed_tensors) != len(refilled_tensors) or any(
        previous is not current
        for previous, current in zip(
            managed_tensors,
            refilled_tensors,
            strict=True,
        )
    ):
        raise RuntimeError(
            "FSDP compute-representation refill replaced managed storage"
        )


class _BuildFSDPComputeWeight(torch.autograd.Function):
    """Bridge compute-weight gradients for GraphTrainer's SimpleFSDP.

    FSDP2 creates this gradient edge internally for its post-all-gather output.
    GraphTrainer's SimpleFSDP builds the compute weight outside FSDP2, so this
    function routes its logical weight gradient back through the BF16 gather.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx, weight: torch.Tensor, wrapper: _FSDPWeightWithComputeRepresentation
    ):
        del ctx
        return wrapper._build_compute_weight(weight)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_weight: torch.Tensor):
        del ctx
        return grad_weight, None


class _FSDPWeightWithComputeRepresentation(torch.Tensor):
    """High-precision FSDP storage with unshard-lifetime compute operands."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, *args: Any, **kwargs: Any):
        del args
        logical_size = kwargs.get("_logical_size", tensor.size())
        logical_stride = kwargs.get("_logical_stride", tensor.stride())
        logical_storage_offset = kwargs.get(
            "_logical_storage_offset",
            tensor.storage_offset(),
        )
        logical_dtype = kwargs.get("_logical_dtype", tensor.dtype)
        logical_device = kwargs.get("_logical_device", tensor.device)
        logical_requires_grad = kwargs.get(
            "_logical_requires_grad",
            tensor.requires_grad,
        )
        return torch.Tensor._make_wrapper_subclass(
            cls,
            logical_size,
            strides=logical_stride,
            storage_offset=logical_storage_offset,
            dtype=logical_dtype,
            layout=tensor.layout,
            device=logical_device,
            pin_memory=tensor.is_pinned(),
            requires_grad=logical_requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        compute_representation: Any = None,
        **logical_metadata: Any,
    ) -> None:
        del logical_metadata
        self._tensor = tensor if compute_representation is None else None
        self._compute_representation = compute_representation

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        del types
        template = None
        preserve_wrapper = func in _FSDP_SUBCLASS_OPS or func in (
            torch.ops.aten.detach.default,
            torch.ops.aten.alias.default,
        )

        def unwrap(tensor: _FSDPWeightWithComputeRepresentation) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif preserve_wrapper and not tensor._same_metadata(template):
                raise RuntimeError(
                    "FSDP operation mixed compute weight representations"
                )
            if tensor._tensor is None:
                return torch.empty_strided(
                    tensor.size(),
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device="meta",
                    requires_grad=tensor.requires_grad,
                )
            return tensor._tensor

        def wrap_sharded_output(tensor: torch.Tensor):
            assert template is not None
            return type(template)(tensor, template._compute_representation)

        def wrap_compute_view(tensor: torch.Tensor):
            assert template is not None
            compute_representation = template._compute_representation
            assert compute_representation is not None
            anchor = template._fsdp_managed_tensors(compute_representation)[0]
            return type(template)(
                anchor,
                compute_representation,
                _logical_size=tensor.size(),
                _logical_stride=tensor.stride(),
                _logical_storage_offset=tensor.storage_offset(),
                _logical_dtype=template.dtype,
                _logical_device=template.device,
                _logical_requires_grad=tensor.requires_grad,
            )

        original_args = args
        original_kwargs = kwargs or {}
        args, kwargs = pytree.tree_map_only(
            cls,
            unwrap,
            (original_args, original_kwargs),
        )
        if template is not None and template._tensor is None:
            if func in _FSDP_COMPUTE_FACTORY_OPS:
                kwargs["device"] = template.device
                return func(*args, **kwargs)
            if func not in _FSDP_COMPUTE_VIEW_OPS:
                raise RuntimeError(
                    f"{func} attempted to read a storage-free FSDP compute weight"
                )
            output = func(*args, **kwargs)
            wrapped = pytree.tree_map_only(
                torch.Tensor,
                wrap_compute_view,
                output,
            )
            return return_and_correct_aliasing(
                func,
                original_args,
                original_kwargs,
                wrapped,
            )
        output = func(*args, **kwargs)
        if not preserve_wrapper:
            return output
        assert template is not None
        return pytree.tree_map_only(
            torch.Tensor,
            wrap_sharded_output,
            output,
        )

    def _same_metadata(self, other: _FSDPWeightWithComputeRepresentation) -> bool:
        return (
            type(self) is type(other)
            and self._compute_representation is other._compute_representation
        )

    def _build_compute_representation(
        self,
        logical_weight: torch.Tensor,
        out: Any = None,
    ) -> Any:
        raise NotImplementedError

    def _fsdp_managed_tensors(
        self,
        compute_representation: Any,
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _build_compute_weight(self, weight: torch.Tensor):
        local_weight = weight._local_tensor if isinstance(weight, DTensor) else weight
        source = local_weight
        if isinstance(local_weight, _FSDPWeightWithComputeRepresentation):
            if local_weight._tensor is None:
                raise RuntimeError("FSDP weight already has a compute representation")
            source = local_weight._tensor
        with torch.no_grad():
            compute_representation = self._build_compute_representation(source)
        compute_local = type(self)(local_weight, compute_representation)
        if not isinstance(weight, DTensor):
            return compute_local
        return DTensor.from_local(
            compute_local,
            weight.device_mesh,
            weight.placements,
            run_check=False,
            shape=weight.shape,
            stride=weight.stride(),
        )

    def build_compute_weight(self, weight: torch.Tensor):
        """Build a storage-free compute parameter from a SimpleFSDP weight."""
        return _BuildFSDPComputeWeight.apply(weight, self)

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self) -> bool:
        """Release the high-precision all-gather output after state construction."""
        return True

    def fsdp_pre_all_gather(
        self,
        mesh,
        outer_size,
        outer_stride,
        module,
        mp_policy,
    ):
        """Return the high-precision communication tensor."""
        del outer_stride, module
        if self._tensor is None:
            raise RuntimeError("Cannot all-gather an unsharded FSDP weight")
        if outer_size[0] % mesh.size() != 0:
            raise ValueError(
                "FSDP compute weights require dimension 0 to be evenly divisible "
                "by the FSDP shard mesh size"
            )
        source = self._tensor
        dtype = mp_policy.param_dtype or source.dtype
        return (source.to(dtype),), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs,
        metadata,
        param_dtype,
        *,
        out=None,
    ):
        """Create or refill the compute weight representation after all-gather."""
        del metadata, param_dtype
        (gathered_weight,) = all_gather_outputs

        # On the first unshard, FSDP has no compute-weight container or managed
        # tensors yet. Build both and return them to FSDP. With RAF=False, FSDP
        # keeps this representation alive through forward and backward.
        if out is None:
            with torch.no_grad():
                compute_representation = self._build_compute_representation(
                    gathered_weight
                )
            compute_weight = type(self)(gathered_weight, compute_representation)
            return compute_weight, self._fsdp_managed_tensors(compute_representation)

        # After FSDP releases and later unshards the weight again, ``out`` is
        # the same compute-weight object returned above. This occurs between
        # forward and backward with RAF=True, or after a later reshard. Refill
        # the same managed tensor objects so existing module and autograd
        # references remain valid.
        target = out._local_tensor if isinstance(out, DTensor) else out
        if not isinstance(target, type(self)):
            raise RuntimeError("FSDP output does not own a compute representation")
        existing_compute_representation = target.compute_representation
        if existing_compute_representation is None:
            raise RuntimeError("FSDP output does not own a compute representation")
        managed_tensors = self._fsdp_managed_tensors(existing_compute_representation)
        with (
            torch.no_grad(),
            # Refilling lifecycle-managed storage is not a user-visible tensor
            # mutation and must not invalidate saved-tensor version checks.
            torch.autograd._unsafe_preserve_version_counter(managed_tensors),
        ):
            refilled_compute_representation = self._build_compute_representation(
                gathered_weight,
                out=existing_compute_representation,
            )
        refilled_tensors = self._fsdp_managed_tensors(refilled_compute_representation)
        _validate_refilled_tensor_identity(managed_tensors, refilled_tensors)
        target._compute_representation = refilled_compute_representation

    @property
    def compute_representation(self) -> Any:
        """Return the representation for the current unshard lifetime."""
        return self._compute_representation


__all__ = ["_FSDPWeightWithComputeRepresentation"]
