import torch


def gradient_clipping(parameters, max_l2_norm: float):
    total = 0
    for p in parameters:
        if p.grad is not None:
            total += torch.sum(p.grad ** 2)
    grad_norm = torch.sqrt(total)
    if grad_norm < max_l2_norm:
        return parameters
    scale = max_l2_norm / (grad_norm + 1e-6)
    for p in parameters:
        if p.grad is not None:
            p.grad *= scale
    return parameters


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay,
                    "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    step = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    grad = p.grad

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    step_size = lr / bias_correction1

                    if weight_decay:
                        p.mul_(1 - lr * weight_decay)

                    denom = exp_avg_sq.sqrt()
                    denom.div_(bias_correction2 ** 0.5).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
