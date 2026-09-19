from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.train_difnet_v0b import _assert_forward_finite, _guarded_optimizer_step, _read_jsonl


class CountingSgd(torch.optim.SGD):
    step_count: int

    def __init__(self, params, lr: float = 0.1) -> None:  # type: ignore[no-untyped-def]
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):  # type: ignore[no-untyped-def]
        self.step_count += 1
        return super().step(closure)


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    out_dir = tmp_path / "run"
    events_path = out_dir / "events.jsonl"
    return out_dir, events_path


def test_forward_non_finite_loss_writes_failure_before_backward(tmp_path: Path) -> None:
    out_dir, events_path = _paths(tmp_path)
    pred = torch.ones(1, 1, 4)
    loss = torch.tensor(float("nan"))

    with pytest.raises(RuntimeError, match="non-finite training failure"):
        _assert_forward_finite(
            pred=pred,
            loss=loss,
            out_dir=out_dir,
            events_path=events_path,
            epoch=1,
            step=2,
            optimizer_step=3,
            valid_ratio=1.0,
        )

    events = _read_jsonl(events_path)
    assert events[-1]["event"] == "non_finite_training_failure"
    assert events[-1]["reason"] == "non_finite_forward_or_loss"
    assert Path(events[-1]["failure_report"]).exists()


def test_guarded_optimizer_step_blocks_non_finite_gradient(tmp_path: Path) -> None:
    out_dir, events_path = _paths(tmp_path)
    model = torch.nn.Linear(2, 1)
    optimizer = CountingSgd(model.parameters())
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    model.bias.grad = torch.zeros_like(model.bias)

    with pytest.raises(RuntimeError, match="non-finite training failure"):
        _guarded_optimizer_step(
            model=model,
            optimizer=optimizer,
            out_dir=out_dir,
            events_path=events_path,
            epoch=1,
            step=2,
            optimizer_step=3,
            loss_value=0.5,
            valid_ratio=1.0,
        )

    events = _read_jsonl(events_path)
    assert optimizer.step_count == 0
    assert events[-1]["reason"] == "non_finite_gradient"
    assert events[-1]["details"]["gradient"]["first_bad_parameter"] == "weight"
    assert Path(events[-1]["failure_report"]).exists()


def test_guarded_optimizer_step_allows_finite_gradient(tmp_path: Path) -> None:
    out_dir, events_path = _paths(tmp_path)
    model = torch.nn.Linear(2, 1)
    optimizer = CountingSgd(model.parameters())
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()

    grad_norm = _guarded_optimizer_step(
        model=model,
        optimizer=optimizer,
        out_dir=out_dir,
        events_path=events_path,
        epoch=1,
        step=2,
        optimizer_step=3,
        loss_value=float(loss.item()),
        valid_ratio=1.0,
    )

    assert optimizer.step_count == 1
    assert grad_norm >= 0.0
    assert not events_path.exists()
