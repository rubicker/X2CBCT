from __future__ import annotations

import pytest
import torch

from x2cbct_difnet.official import synthetic_difnet_item
from x2cbct_difnet.v2_hash_model import (
    C2RV_COMBINE,
    C2RV_MODEL_VERSION,
    DUFAL_COMBINE,
    DUFAL_MODEL_VERSION,
    HASH_MODEL_VERSIONS,
    CudaHashGridEncoder,
    HashGridConfig,
    validate_capsule_kwargs_for_combine,
    build_difnet_model,
    model_config,
    resolve_checkpoint_capsule_kwargs,
    resolve_checkpoint_combine,
    resolve_checkpoint_hash_backend,
    resolve_checkpoint_mid_channels,
    resolve_checkpoint_model_version,
    resolve_checkpoint_view_count,
)


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_default_model_builds_official_baseline_without_hashgrid_backend() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version="v0b",
        num_views=2,
        combine="mlp",
        mid_channels=8,
        hashgrid_backend="cuda",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=8)

    with torch.no_grad():
        pred, gt = model(item)

    assert pred.shape == (1, 1, 8)
    assert gt.shape == (1, 1, 8)
    assert not hasattr(model, "position_encoder")


@pytest.mark.parametrize("method", HASH_MODEL_VERSIONS)
def test_v2_hash_methods_run_synthetic_forward_with_torch_backend(method: str) -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=method,
        num_views=2,
        combine="mlp",
        mid_channels=8,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=8)

    with torch.no_grad():
        pred, gt = model(item)

    config = model_config(model, {})
    assert pred.shape == (1, 1, 8)
    assert gt.shape == (1, 1, 8)
    assert config["model_version"] == method
    assert config["hashgrid"]["backend_actual"] == "torch"
    assert config["hashgrid"]["output_dim"] == 32


def test_v0b_capsule_a1_runs_synthetic_forward_with_angles_and_mask() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version="v0b",
        num_views=2,
        combine="capsule-a1",
        mid_channels=8,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=8)

    with torch.no_grad():
        pred, gt = model(item)

    config = model_config(model, {})
    assert pred.shape == (1, 1, 8)
    assert gt.shape == (1, 1, 8)
    assert config["model_version"] == "v0b"
    assert config["combine"] == "capsule-a1"
    assert config["capsule"]["hidden_dim"] == 8
    assert "hashgrid" not in config


def test_v2_hash_capsule_a1_runs_synthetic_forward_with_torch_backend() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version="v2-hash-a2",
        num_views=2,
        combine="capsule-a1",
        mid_channels=8,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=8)

    with torch.no_grad():
        pred, gt = model(item)

    config = model_config(model, {})
    assert pred.shape == (1, 1, 8)
    assert gt.shape == (1, 1, 8)
    assert config["model_version"] == "v2-hash-a2"
    assert config["combine"] == "capsule-a1"
    assert config["hashgrid"]["backend_actual"] == "torch"
    assert config["capsule"]["layers"] == 2


def test_c2rv_style_runs_synthetic_forward_with_grid_keys() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=C2RV_MODEL_VERSION,
        num_views=2,
        combine=C2RV_COMBINE,
        mid_channels=8,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=8)

    with torch.no_grad():
        pred, gt = model(item)

    config = model_config(model, {})
    assert pred.shape == (1, 1, 8)
    assert gt.shape == (1, 1, 8)
    assert config["model_version"] == C2RV_MODEL_VERSION
    assert config["combine"] == C2RV_COMBINE
    assert config["c2rv"]["grid_scales"] == [16, 8, 4]


def test_c2rv_style_reports_missing_grid_payload() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=C2RV_MODEL_VERSION,
        num_views=2,
        combine=C2RV_COMBINE,
        mid_channels=8,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=8)
    item.pop("c2rv_grid_proj_points_s0")

    with pytest.raises(ValueError, match="c2rv_grid_proj_points_s0"):
        model(item)


@pytest.mark.parametrize("num_views", [2, 5, 10])
def test_dufal_style_runs_synthetic_forward_with_max_view_fusion(num_views: int) -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=DUFAL_MODEL_VERSION,
        num_views=num_views,
        combine=DUFAL_COMBINE,
        mid_channels=8,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=num_views, image_size=32, num_points=8)

    with torch.no_grad():
        pred, gt = model(item)

    config = model_config(model, {})
    assert pred.shape == (1, 1, 8)
    assert gt.shape == (1, 1, 8)
    assert config["model_version"] == DUFAL_MODEL_VERSION
    assert config["combine"] == DUFAL_COMBINE
    assert config["dufal"]["view_fusion"] == "max"
    assert "c2rv" not in config


def test_dufal_style_runs_without_c2rv_grid_or_valid_mask() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=DUFAL_MODEL_VERSION,
        num_views=2,
        combine=DUFAL_COMBINE,
        mid_channels=8,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=8)
    for key in list(item):
        if key.startswith("c2rv_grid_"):
            item.pop(key)
    item.pop("valid_mask")

    with torch.no_grad():
        pred, gt = model(item)

    assert pred.shape == (1, 1, 8)
    assert gt.shape == (1, 1, 8)


def test_dufal_frequency_components_run_backward() -> None:
    from x2cbct_difnet.dufal_model import DuFalProjectionFeatureExtractor

    extractor = DuFalProjectionFeatureExtractor(in_channels=1, feature_channels=8, blocks=1, modes=64, patch_size=7)
    x = torch.rand((2, 1, 31, 29), requires_grad=True)

    out = extractor(x)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == (2, 8, 31, 29)
    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dufal_style_requires_matching_combine() -> None:
    with pytest.raises(ValueError, match="requires combine=dufal"):
        build_difnet_model(
            PROJECT_ROOT,
            model_version=DUFAL_MODEL_VERSION,
            num_views=2,
            combine="max",
            mid_channels=8,
            hashgrid_backend="torch",
        )
    with pytest.raises(ValueError, match="requires model_version=dufal-style"):
        build_difnet_model(
            PROJECT_ROOT,
            model_version="v0b",
            num_views=2,
            combine=DUFAL_COMBINE,
            mid_channels=8,
            hashgrid_backend="torch",
        )


def test_v0b_gaze_sar_hu_runs_without_angles_or_valid_mask() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version="v0b",
        num_views=2,
        combine="gaze_sar_hu",
        mid_channels=128,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=4)
    item.pop("view_angles_deg")
    item.pop("valid_mask")

    with torch.no_grad():
        pred, gt = model(item)

    config = model_config(model, {})
    assert pred.shape == (1, 1, 4)
    assert gt.shape == (1, 1, 4)
    assert config["combine"] == "gaze_sar_hu"
    assert config["capsule_routing"]["readout"] == "direct_scalar"
    assert config["capsule_routing"]["uses_valid_mask"] is False


@pytest.mark.parametrize(
    ("combine", "mid_channels", "expected_primary_dim"),
    [
        ("eff1_sar_mlp", 32, 4),
        ("eff2_sar_mlp", 64, 4),
        ("gaze_sar_hu", 128, 8),
    ],
)
def test_v0b_routed_capsule_mid_channel_plans_run_synthetic_forward(
    combine: str,
    mid_channels: int,
    expected_primary_dim: int,
) -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version="v0b",
        num_views=2,
        combine=combine,
        mid_channels=mid_channels,
        hashgrid_backend="torch",
    )
    item = synthetic_difnet_item(num_views=2, image_size=32, num_points=4)

    with torch.no_grad():
        pred, gt = model(item)

    config = model_config(model, {})
    assert pred.shape == (1, 1, 4)
    assert gt.shape == (1, 1, 4)
    assert config["capsule_routing"]["feature_channels"] == mid_channels
    assert config["capsule_routing"]["primary_capsule_dim"] == expected_primary_dim


def test_efficient_and_gaze_routes_are_v0b_mid_channels_32_64_128_only() -> None:
    with pytest.raises(ValueError, match="mid_channels 32, 64, or 128"):
        build_difnet_model(
            PROJECT_ROOT,
            model_version="v0b",
            num_views=2,
            combine="eff1_flat",
            mid_channels=8,
            hashgrid_backend="torch",
        )
    with pytest.raises(ValueError, match="model_version=v0b"):
        build_difnet_model(
            PROJECT_ROOT,
            model_version="v2-hash-a1",
            num_views=2,
            combine="gaze_sar_mlp",
            mid_channels=128,
            hashgrid_backend="torch",
        )


def test_efficient_and_gaze_routes_reject_legacy_capsule_arguments() -> None:
    with pytest.raises(ValueError, match="does not support legacy --capsule"):
        validate_capsule_kwargs_for_combine(
            "eff1_flat",
            capsule_hidden_dim=128,
            capsule_layers=None,
            capsule_heads=None,
            capsule_mlp_ratio=None,
            capsule_dropout=None,
            capsule_valid_mask=None,
        )
    validate_capsule_kwargs_for_combine(
        "capsule-a1",
        capsule_hidden_dim=128,
        capsule_layers=None,
        capsule_heads=None,
        capsule_mlp_ratio=None,
        capsule_dropout=None,
        capsule_valid_mask=True,
    )


def test_a1_uses_160_classifier_input_channels() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version="v2-hash-a1",
        num_views=2,
        combine="mlp",
        mid_channels=8,
        hashgrid_backend="torch",
    )

    config = model_config(model, {})

    assert config["position_injection"]["classifier_input_channels"] == 40


def test_a2_keeps_mid_channel_classifier_input() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version="v2-hash-a2",
        num_views=2,
        combine="mlp",
        mid_channels=8,
        hashgrid_backend="torch",
    )

    config = model_config(model, {})

    assert config["position_injection"]["classifier_input_channels"] == 8


def test_auto_hashgrid_backend_uses_torch_without_cuda_preference() -> None:
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version="v2-hash-a1",
        num_views=2,
        combine="mlp",
        mid_channels=8,
        hashgrid_backend="auto",
        prefer_cuda_hashgrid=False,
    )

    assert model_config(model, {})["hashgrid"]["backend_actual"] == "torch"


def test_checkpoint_model_version_resolves_v2_config() -> None:
    checkpoint = {"config": {"model_version": "v2-hash-a2b"}}

    assert resolve_checkpoint_model_version(checkpoint) == "v2-hash-a2b"
    assert resolve_checkpoint_model_version({}) == "v0b"


def test_checkpoint_combine_prefers_model_config() -> None:
    checkpoint = {"config": {"combine": "mlp", "model_config": {"combine": "capsule-a1"}}}

    assert resolve_checkpoint_combine(checkpoint) == "capsule-a1"
    assert resolve_checkpoint_combine({}) == "mlp"


def test_checkpoint_mid_channels_prefers_model_config() -> None:
    checkpoint = {"config": {"mid_channels": 64, "model_config": {"mid_channels": 32}}}

    assert resolve_checkpoint_mid_channels(checkpoint) == 32
    assert resolve_checkpoint_mid_channels({"config": {"mid_channels": 64}}) == 64
    assert resolve_checkpoint_mid_channels({}, requested=16) == 16
    assert resolve_checkpoint_mid_channels({}) == 128


def test_checkpoint_view_count_prefers_model_config() -> None:
    checkpoint = {"config": {"view_count": 5, "model_config": {"num_views": 2}}}

    assert resolve_checkpoint_view_count(checkpoint) == 2
    assert resolve_checkpoint_view_count({"config": {"view_count": 5}}) == 5
    assert resolve_checkpoint_view_count({}) is None


def test_checkpoint_resolves_c2rv_style_model_config() -> None:
    checkpoint = {
        "config": {
            "model_config": {
                "model_version": C2RV_MODEL_VERSION,
                "combine": C2RV_COMBINE,
            }
        }
    }

    assert resolve_checkpoint_model_version(checkpoint) == C2RV_MODEL_VERSION
    assert resolve_checkpoint_combine(checkpoint) == C2RV_COMBINE


def test_checkpoint_resolves_dufal_style_model_config() -> None:
    checkpoint = {
        "config": {
            "model_config": {
                "model_version": DUFAL_MODEL_VERSION,
                "combine": DUFAL_COMBINE,
                "dufal": {"view_fusion": "max"},
            }
        }
    }

    assert resolve_checkpoint_model_version(checkpoint) == DUFAL_MODEL_VERSION
    assert resolve_checkpoint_combine(checkpoint) == DUFAL_COMBINE


def test_checkpoint_capsule_kwargs_prefers_model_config_defaults() -> None:
    checkpoint = {
        "config": {
            "model_config": {
                "capsule": {
                    "hidden_dim": 256,
                    "layers": 3,
                    "heads": 8,
                    "mlp_ratio": 3.0,
                    "dropout": 0.1,
                    "valid_mask": True,
                }
            }
        }
    }

    assert resolve_checkpoint_capsule_kwargs(checkpoint)["capsule_hidden_dim"] == 256
    assert resolve_checkpoint_capsule_kwargs(checkpoint)["capsule_layers"] == 3
    assert resolve_checkpoint_capsule_kwargs(checkpoint)["capsule_valid_mask"] is True


def test_checkpoint_hash_backend_prefers_actual_model_config_backend() -> None:
    checkpoint = {
        "config": {
            "hashgrid_backend": "auto",
            "model_config": {"hashgrid": {"backend_requested": "auto", "backend_actual": "torch"}},
        }
    }

    assert resolve_checkpoint_hash_backend(checkpoint) == "torch"


def test_cuda_hashgrid_backend_fails_before_import_when_cuda_is_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        CudaHashGridEncoder(PROJECT_ROOT, HashGridConfig())
