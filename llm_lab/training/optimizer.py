import torch


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
                    step = state.get("step")
                    exp_avg = state.get("exp_avg")
                    exp_avg_sq = state.get("exp_avg_sq")
                    grad = p.grad
                    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad**2
                    step += 1
                    exp_avg_hat = exp_avg / (1 - beta1 ** step)
                    exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** step)
                    p -= lr * \
                        ((exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + eps)) +
                         weight_decay * p)
                    state["step"] = step
                    state["exp_avg"] = exp_avg
                    state["exp_avg_sq"] = exp_avg_sq
        return loss
