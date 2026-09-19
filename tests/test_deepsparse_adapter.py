from __future__ import annotations

import torch

from x2cbct_deepsparse.model import masked_query_view_feats


def test_masked_query_view_feats_honors_per_point_mask() -> None:
    # Constant feature maps make the expected max independent of coordinates.
    view_feats = torch.tensor([[[[[1.0]]], [[[9.0]]]]])
    points_proj = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    valid = torch.tensor([[[True, False], [False, True]]])

    pooled = masked_query_view_feats(view_feats, points_proj, view_mask=valid)

    assert pooled.shape == (1, 1, 2)
    assert torch.equal(pooled, torch.tensor([[[1.0, 9.0]]]))


def test_masked_query_view_feats_zeroes_all_invalid_points() -> None:
    view_feats = torch.tensor([[[[[1.0]]], [[[9.0]]]]])
    points_proj = torch.zeros((1, 2, 1, 2), dtype=torch.float32)
    valid = torch.zeros((1, 2, 1), dtype=torch.bool)

    pooled = masked_query_view_feats(view_feats, points_proj, view_mask=valid)

    assert torch.equal(pooled, torch.zeros((1, 1, 1)))
    assert torch.isfinite(pooled).all()


def test_masked_query_view_feats_keeps_official_view_mask_contract() -> None:
    view_feats = torch.tensor([[[[[2.0]]], [[[7.0]]]]])
    points_proj = torch.zeros((1, 2, 3, 2), dtype=torch.float32)
    view_mask = torch.tensor([[True, False]])

    pooled = masked_query_view_feats(view_feats, points_proj, view_mask=view_mask)

    assert torch.equal(pooled, torch.full((1, 1, 3), 2.0))
