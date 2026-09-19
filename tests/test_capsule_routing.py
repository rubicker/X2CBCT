from __future__ import annotations

import pytest
import torch

from x2cbct_difnet.capsule_routing import (
    CAPSULE_SQUASH_MODE_CHOICES,
    EfficientCapsuleFusion,
    EfficientCapsuleRoutingLayer,
    GazeSarFusion,
    GazeSarLayer,
    GazeSarReadout,
    efficient_squash,
    get_capsule_squash_mode,
    hinton_squash,
    legacy_efficient_squash,
    set_capsule_squash_mode,
    stable_efficient_squash,
)


@pytest.fixture(autouse=True)
def _stable_capsule_squash_mode() -> None:
    set_capsule_squash_mode("stable")
    yield
    set_capsule_squash_mode("stable")


def test_efficient_squash_keeps_capsule_length_below_one() -> None:
    x = torch.randn(2, 3, 5, 8)

    y = efficient_squash(x)

    assert y.shape == x.shape
    assert torch.linalg.vector_norm(y, dim=-1).max().item() <= 1.0


@pytest.mark.parametrize("norm_target", [90.0, 100.0, 200.0])
def test_stable_efficient_squash_backward_stays_finite_at_large_norm(norm_target: float) -> None:
    x = torch.full((2, 4), norm_target / 2.0, dtype=torch.float32, requires_grad=True)

    y = efficient_squash(x)
    y.sum().backward()

    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()


def test_stable_efficient_squash_handles_zero_norm() -> None:
    x = torch.zeros(2, 4, dtype=torch.float32, requires_grad=True)

    y = efficient_squash(x)
    y.sum().backward()

    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()


def test_legacy_efficient_squash_reproduces_large_norm_backward_risk() -> None:
    x = torch.full((1, 4), 50.0, dtype=torch.float32, requires_grad=True)

    y = legacy_efficient_squash(x)
    y.sum().backward()

    assert torch.isfinite(y).all()
    assert not torch.isfinite(x.grad).all()


def test_hinton_squash_backward_stays_finite_at_large_norm() -> None:
    x = torch.full((1, 4), 50.0, dtype=torch.float32, requires_grad=True)

    y = hinton_squash(x)
    y.sum().backward()

    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    assert torch.linalg.vector_norm(y, dim=-1).max().item() <= 1.0


def test_capsule_squash_mode_switches_runtime_behavior() -> None:
    assert set(CAPSULE_SQUASH_MODE_CHOICES) == {"stable", "legacy", "hinton"}
    set_capsule_squash_mode("hinton")

    assert get_capsule_squash_mode() == "hinton"

    with pytest.raises(ValueError, match="unsupported capsule squash mode"):
        set_capsule_squash_mode("bad-mode")


def test_stable_fusion_backward_keeps_upstream_features_finite_under_large_primary_norm() -> None:
    fusion = EfficientCapsuleFusion(feature_channels=64, num_views=5, combine="eff1_sar_mlp")
    p_feats = torch.full((1, 64, 4, 5), 50.0, dtype=torch.float32, requires_grad=True)

    loss = fusion(p_feats).square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert p_feats.grad is not None
    assert torch.isfinite(p_feats.grad).all()


def test_legacy_fusion_backward_can_pollute_upstream_feature_gradients() -> None:
    set_capsule_squash_mode("legacy")
    fusion = EfficientCapsuleFusion(feature_channels=64, num_views=5, combine="eff1_sar_mlp")
    p_feats = torch.full((1, 64, 4, 5), 50.0, dtype=torch.float32, requires_grad=True)

    loss = fusion(p_feats).square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert p_feats.grad is not None
    assert not torch.isfinite(p_feats.grad).all()


def test_efficient_routing_layer_outputs_parent_capsules_and_normalized_coupling() -> None:
    layer = EfficientCapsuleRoutingLayer(
        input_capsules=32,
        input_dim=8,
        output_capsules=16,
        output_dim=16,
    )
    x = torch.randn(2, 7, 32, 8)

    y = layer(x)
    coupling = layer.coupling_coefficients(x)

    assert y.shape == (2, 7, 16, 16)
    assert coupling.shape == (2, 7, 16, 32, 1)
    assert torch.allclose(coupling.sum(dim=2), torch.ones_like(coupling.sum(dim=2)), atol=1e-6)
    assert dict(layer.named_parameters())["b"].shape == (16, 32, 1)


def test_efficient_routing_layer_reuses_votes_for_coupling() -> None:
    layer = EfficientCapsuleRoutingLayer(
        input_capsules=4,
        input_dim=3,
        output_capsules=2,
        output_dim=5,
    )
    x = torch.randn(2, 3, 4, 3)

    votes = layer.votes(x)

    assert torch.allclose(layer.coupling_coefficients_from_votes(votes), layer.coupling_coefficients(x))


def test_efficient_routing_layer_forward_computes_votes_once() -> None:
    class CountingEfficientCapsuleRoutingLayer(EfficientCapsuleRoutingLayer):
        vote_call_count: int

        def __init__(self) -> None:
            super().__init__(
                input_capsules=4,
                input_dim=3,
                output_capsules=2,
                output_dim=5,
            )
            self.vote_call_count = 0

        def votes(self, x: torch.Tensor) -> torch.Tensor:
            self.vote_call_count += 1
            return super().votes(x)

    layer = CountingEfficientCapsuleRoutingLayer()
    x = torch.randn(2, 3, 4, 3)

    layer(x)

    assert layer.vote_call_count == 1


def test_gaze_sar_layer_and_readout_shapes() -> None:
    x = torch.randn(2, 7, 32, 8)
    layer = GazeSarLayer(input_dim=8, output_capsules=16, output_dim=16)
    readout_128 = GazeSarReadout(input_dim=16, output_dim=128)
    readout_1 = GazeSarReadout(input_dim=16, output_dim=1)

    hidden = layer(x)

    assert hidden.shape == (2, 7, 16, 16)
    assert readout_128(hidden).shape == (2, 7, 128)
    assert readout_1(hidden).shape == (2, 7, 1)


def test_fusion_modules_follow_prd_shapes_and_ignore_auxiliary_fields() -> None:
    p_feats = torch.randn(2, 64, 7, 2)
    valid_mask = torch.zeros(2, 2, 7, dtype=torch.bool)
    view_angles_deg = torch.randn(2, 2)

    eff = EfficientCapsuleFusion(feature_channels=64, num_views=2, combine="eff2_sar_mlp")
    gaze = GazeSarFusion(feature_channels=64, num_views=2, combine="gaze_sar_hu")

    eff_out = eff(p_feats, valid_mask=valid_mask, view_angles_deg=view_angles_deg)
    gaze_out = gaze(p_feats, valid_mask=valid_mask, view_angles_deg=view_angles_deg)

    assert eff.output_kind == "feature_128"
    assert eff_out.shape == (2, 128, 7)
    assert gaze.output_kind == "direct_scalar"
    assert gaze_out.shape == (2, 1, 7)
    assert eff.info()["uses_valid_mask"] is False
    assert gaze.info()["uses_geometry_tokens"] is False


@pytest.mark.parametrize(
    ("combine", "feature_channels", "num_views", "expected_channels", "expected_kind", "expected_input_capsules"),
    [
        ("eff1_flat", 32, 5, 160, "flat_feature", 40),
        ("eff2_flat", 32, 5, 160, "flat_feature", 40),
        ("eff1_flat", 64, 5, 320, "flat_feature", 80),
        ("eff2_flat", 64, 5, 320, "flat_feature", 80),
        ("eff1_flat", 128, 2, 256, "flat_feature", 32),
        ("eff1_flat", 128, 5, 640, "flat_feature", 80),
        ("eff1_flat", 128, 10, 1280, "flat_feature", 160),
        ("eff2_flat", 128, 2, 256, "flat_feature", 32),
        ("eff2_flat", 128, 5, 640, "flat_feature", 80),
        ("eff2_flat", 128, 10, 1280, "flat_feature", 160),
        ("eff1_sar_mlp", 32, 2, 128, "feature_128", 16),
        ("eff2_sar_mlp", 64, 5, 128, "feature_128", 80),
        ("eff1_sar_hu", 64, 2, 1, "direct_scalar", 32),
        ("eff2_sar_hu", 128, 10, 1, "direct_scalar", 160),
        ("gaze_sar_mlp", 32, 5, 128, "feature_128", 40),
        ("gaze_sar_hu", 64, 10, 1, "direct_scalar", 160),
    ],
)
def test_all_new_fusion_combinations_follow_output_contract(
    combine: str,
    feature_channels: int,
    num_views: int,
    expected_channels: int,
    expected_kind: str,
    expected_input_capsules: int,
) -> None:
    p_feats = torch.randn(1, feature_channels, 3, num_views)
    if combine.startswith("eff"):
        fusion = EfficientCapsuleFusion(feature_channels=feature_channels, num_views=num_views, combine=combine)
    else:
        fusion = GazeSarFusion(feature_channels=feature_channels, num_views=num_views, combine=combine)

    out = fusion(p_feats)

    assert out.shape == (1, expected_channels, 3)
    assert fusion.output_kind == expected_kind
    assert fusion.info()["input_capsules"] == expected_input_capsules


def test_efficient_fusion_rejects_unsupported_feature_channels() -> None:
    with pytest.raises(ValueError, match="supports feature_channels 32, 64, or 128"):
        EfficientCapsuleFusion(feature_channels=96, num_views=5, combine="eff1_sar_mlp")
