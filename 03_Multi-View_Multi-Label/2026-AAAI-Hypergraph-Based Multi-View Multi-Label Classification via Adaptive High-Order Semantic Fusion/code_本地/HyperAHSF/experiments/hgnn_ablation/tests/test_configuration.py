from __future__ import annotations

from importlib import import_module

import pytest


def _configuration_api():
    try:
        module = import_module("experiments.hgnn_ablation.configuration")
    except ModuleNotFoundError as exc:
        pytest.fail(f"common ablation configuration is missing: {exc}")
    return module.AblationFlags, module.validate_ablation_flags


def test_default_flags_disable_every_mechanism() -> None:
    AblationFlags, validate_ablation_flags = _configuration_api()
    flags = AblationFlags()

    assert flags.enabled_count == 0
    validate_ablation_flags(flags)


@pytest.mark.parametrize(
    "flag_kwargs",
    [
        {"use_residual": True, "use_attention": True},
        {"use_residual": True, "learnable_hyperedge_weight": True},
        {"use_attention": True, "learnable_hyperedge_weight": True},
        {
            "use_residual": True,
            "learnable_hyperedge_weight": True,
            "use_attention": True,
        },
    ],
)
def test_multiple_mechanisms_raise_chinese_error(flag_kwargs: dict[str, bool]) -> None:
    AblationFlags, validate_ablation_flags = _configuration_api()
    flags = AblationFlags(**flag_kwargs)
    with pytest.raises(ValueError, match="三种机制只能单独启用"):
        validate_ablation_flags(flags)


@pytest.mark.parametrize(
    "flag_kwargs",
    [
        {"use_residual": True},
        {"learnable_hyperedge_weight": True},
        {"use_attention": True},
    ],
)
def test_each_single_mechanism_is_valid(flag_kwargs: dict[str, bool]) -> None:
    AblationFlags, validate_ablation_flags = _configuration_api()
    flags = AblationFlags(**flag_kwargs)
    assert flags.enabled_count == 1
    validate_ablation_flags(flags)
