import torch


def softmax(x: torch.Tensor, dim: int, temperature: float = None):
    if temperature is not None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        x = x / temperature
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)
