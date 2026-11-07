"""FGSM / PGD / multi-restart PGD in [0,1] input space (ℓ∞)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _project(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)


@torch.enable_grad()
def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    x_adv = x.detach().clone().requires_grad_(True)
    loss = loss_fn(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x + eps * grad.sign()).clamp(0.0, 1.0).detach()


@torch.enable_grad()
def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)
    else:
        x_adv = x.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = _project(x_adv + step_size * grad.sign(), x, eps)
    return x_adv.detach()


@torch.enable_grad()
def multi_restart_pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step_size: float,
    steps: int,
    restarts: int,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """Keep per-example adversary with highest loss across restarts."""
    loss_fn = loss_fn or nn.CrossEntropyLoss(reduction="none")
    best_adv = x.clone()
    best_loss = torch.full((x.size(0),), -1e9, device=x.device)

    was_training = model.training
    model.eval()
    for _ in range(restarts):
        adv = pgd(
            model,
            x,
            y,
            eps=eps,
            step_size=step_size,
            steps=steps,
            random_start=True,
            loss_fn=nn.CrossEntropyLoss(),
        )
        with torch.no_grad():
            cur_loss = loss_fn(model(adv), y)
            improve = cur_loss > best_loss
            best_loss = torch.where(improve, cur_loss, best_loss)
            best_adv = torch.where(improve.view(-1, 1, 1, 1), adv, best_adv)
    if was_training:
        model.train()
    return best_adv.detach()


def validate_constraints(
    x: torch.Tensor, x_adv: torch.Tensor, eps: float, tol: float = 1e-5
) -> dict:
    linf = (x_adv - x).abs().reshape(x.size(0), -1).max(dim=1).values
    box_ok = bool((x_adv >= -tol).all() and (x_adv <= 1.0 + tol).all())
    eps_ok = bool((linf <= eps + tol).all())
    return {
        "box_ok": box_ok,
        "eps_ok": eps_ok,
        "max_linf": float(linf.max().item()),
        "mean_linf": float(linf.mean().item()),
    }
