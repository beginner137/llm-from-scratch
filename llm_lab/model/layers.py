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
        # self.bias = torch.zeros(out_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)
