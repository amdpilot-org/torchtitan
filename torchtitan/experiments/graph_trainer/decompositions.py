# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone decomposition passes for GraphTrainer FX graphs."""

from __future__ import annotations

import torch
import torch.fx as fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import selective_decompose

from torchtitan.experiments.graph_trainer.common_utils import _get_graph_modules


def _copy_attr(src: torch.nn.Module, dst: torch.nn.Module, target: str) -> None:
    *prefix, field = target.split(".")
    for name in prefix:
        src_child = getattr(src, name)
        child = getattr(dst, name, None)
        if src_child is child:
            return
        if child is None:
            child = torch.nn.Module()
            setattr(dst, name, child)
        src = src_child
        dst = child

    value = getattr(src, field)
    if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
        persistent = field not in src._non_persistent_buffers_set
        dst.register_buffer(field, value, persistent=persistent)
    else:
        setattr(dst, field, value)


def _apply_decompositions(
    gm: fx.GraphModule,
    example_inputs,
    decomposition_table,
) -> None:
    if example_inputs is None:
        placeholders = gm.graph.find_nodes(op="placeholder")
        if any("val" not in node.meta for node in placeholders):
            return
        example_inputs = tuple(node.meta["val"] for node in placeholders)

    with fx_traceback.preserve_node_meta():
        decomposed = selective_decompose(
            gm,
            *example_inputs,
            decomposition=decomposition_table,
            should_decompose=lambda _node: True,
            trace_joint_graph=False,
        )

    for node in decomposed.graph.find_nodes(op="get_attr"):
        _copy_attr(decomposed, gm, node.target)
    gm.graph = decomposed.graph
    gm.graph.lint()
    gm.recompile()


def apply_decompositions_pass(
    gm: fx.GraphModule,
    example_inputs=None,
    *,
    decomposition_table,
    recurse: bool = False,
    apply_to_root: bool = True,
) -> fx.GraphModule:
    """Apply a decomposition table to the root graph and/or nested subgraphs.

    Args:
        recurse: If ``True``, apply the pass to all nested FX ``GraphModule``
            submodules. The root graph is controlled separately by
            ``apply_to_root``.
    """
    if callable(decomposition_table):
        decomposition_table = decomposition_table()
    if not decomposition_table:
        return gm

    for module in _get_graph_modules(gm, recurse=recurse, apply_to_root=apply_to_root):
        _apply_decompositions(
            module,
            example_inputs if module is gm else None,
            decomposition_table,
        )
    return gm
