# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import call, patch

import pytest

from torchtitan.experiments.rl.batch_invariance import _load_vllm_batch_invariant_bmm


_NEW_MODULE = "vllm.model_executor.determinism.batch_invariant"
_LEGACY_MODULE = "vllm.model_executor.layers.batch_invariant"


@pytest.mark.parametrize(
    ("available_module", "expected_calls"),
    [
        (_NEW_MODULE, [call(_NEW_MODULE)]),
        (_LEGACY_MODULE, [call(_NEW_MODULE), call(_LEGACY_MODULE)]),
    ],
)
def test_load_vllm_batch_invariant_bmm(available_module, expected_calls):
    expected_bmm = object()

    def import_module(module_name):
        if module_name == available_module:
            return SimpleNamespace(bmm_batch_invariant=expected_bmm)
        raise ModuleNotFoundError(name=module_name)

    with patch(
        "torchtitan.experiments.rl.batch_invariance.importlib.import_module",
        side_effect=import_module,
    ) as mock_import:
        assert _load_vllm_batch_invariant_bmm() is expected_bmm

    assert mock_import.call_args_list == expected_calls


def test_load_vllm_batch_invariant_bmm_preserves_dependency_error():
    error = ModuleNotFoundError(name="missing_dependency")

    with (
        patch(
            "torchtitan.experiments.rl.batch_invariance.importlib.import_module",
            side_effect=error,
        ),
        pytest.raises(ModuleNotFoundError) as exc_info,
    ):
        _load_vllm_batch_invariant_bmm()

    assert exc_info.value is error
