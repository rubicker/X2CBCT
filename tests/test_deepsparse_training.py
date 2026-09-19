from __future__ import annotations

import torch

from tools import train_deepsparse


def test_task_loss_supervises_every_sampled_point() -> None:
    prediction = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)

    loss = train_deepsparse._task_loss(prediction, target)
    loss.backward()

    assert torch.equal(loss.detach(), torch.tensor(5.0))
    assert torch.equal(prediction.grad, torch.tensor([[[1.0, 3.0]]]))
