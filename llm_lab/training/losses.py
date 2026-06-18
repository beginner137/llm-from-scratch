import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # -(X_target - X_max) + log(sum(e^(xi - max)))

    # inputs: batch_size vocab_size
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - inputs_max
    # batch_size
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))

    target_logits = shifted.gather(
        dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    losses = logsumexp - target_logits

    return losses.mean()
