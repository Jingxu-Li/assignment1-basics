from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0],
                    "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lamda = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                t += 1  # Increment iteration number
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update weight tensor in-place.
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                lr_t = lr * math.sqrt(1 - beta2**(t)) / (1 - beta1**(t))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr*lamda * p.data
                state["t"] = t  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss
