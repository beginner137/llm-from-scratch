import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        sigma = math.sqrt(2/(in_features + out_features))
        """
        d_in = 3, d_out = 2

        w_1 = [a,b,c], w_2 = [d,e,f]

        Weight matrix:
        [[a,b,c],
        [d,e,f]] -> shape (2,3) = (d_out, d_in)
        """
        weight = torch.empty(out_features, in_features,
                             device=device, dtype=dtype)
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        sigma = 1
        embedding = torch.empty(
            num_embeddings, embedding_dim, device=None, dtype=None)
        self.embedding = nn.Parameter(nn.init.trunc_normal_(
            embedding, mean=0, std=sigma, a=-3, b=3
        ))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1_weight = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3_weight = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2_weight = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1_weight(x)
        w3x = self.w3_weight(x)
        return self.w2_weight((w1x * torch.sigmoid(w1x)) * w3x)
