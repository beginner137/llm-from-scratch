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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_freq = 1.0 / \
            (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        positions = torch.arange(max_seq_len, device=device)
        angles = torch.outer(positions, inv_freq)  # [max_seq_len, d_k // 2]

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_even_rot
        out[..., 1::2] = x_odd_rot
        return out


def softmax(x: torch.Tensor, dim: int):
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = Q @ (K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, with_rope=True, theta=None, max_seq_len=2048, token_positions=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_v = d_model // num_heads
        self.with_rope = with_rope
        self.token_positions = token_positions

        if with_rope:
            self.rope = RotaryPositionalEmbedding(
                theta, self.d_k, max_seq_len, device=device)
        self.q_proj_weight = Linear(
            d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj_weight = Linear(
            d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj_weight = Linear(
            d_model, num_heads * self.d_v, device=device, dtype=dtype)
        self.o_proj_weight = Linear(
            num_heads * self.d_v, d_model, device=device, dtype=dtype)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        sequence_length = in_features.shape[-2]

        Q = self.q_proj_weight(in_features)  # (..., sequence_length, d_model)
        K = self.k_proj_weight(in_features)
        V = self.v_proj_weight(in_features)

        # reshape to (..., sequence_length, num_heads, head_dim)
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.d_k)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.d_v)

        # swap dimension to (..., num_heads, sequence_length, head_dim)
        Q = Q.transpose(-2, -3)
        K = K.transpose(-2, -3)
        V = V.transpose(-2, -3)

        # Apply Rope to Q and K
        if self.with_rope:
            if not self.token_positions:
                self.token_positions = torch.arange(sequence_length)
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)

        # create a lower triangle mask
        causal_mask = torch.tril(torch.ones(
            sequence_length, sequence_length), diagonal=0).bool()

        # (..., num_heads, sequence_length, d_v)
        O = scaled_dot_product_attention(Q, K, V, causal_mask)

        # change to (..., sequence_length, d_v*num_heads)
        O = O.transpose(-2, -3)  # (..., sequence_length,num_heads, d_v)
        O = O.flatten(start_dim=-2)  # (..., sequence_length, d_v*num_heads)

        return self.o_proj_weight(O)  # (..., sequence_length, d_model)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, device=device,
                                       dtype=dtype, with_rope=True, theta=theta, max_seq_len=max_seq_len)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff,
                          device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass RMSNorm
        x1 = self.ln1(x)
        # pass Attention
        x2 = self.attn(x1)
        # residual
        x3 = x + x2

        # RMSNorm
        x4 = self.ln2(x3)
        # pass FFN
        x5 = self.ffn(x4)
        # residual
        return x3 + x5
