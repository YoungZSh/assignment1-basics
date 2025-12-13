import math
import torch
from torch import nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=sigma, 
            a=-3 * sigma, 
            b=3 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3,
            b=3
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))

        nn.init.constant_(self.g, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        g = self.g.to(torch.float32)

        # 计算每个样本沿最后一个维度的均方根
        # square_sum 形状: (...)
        square_sum = einsum(x, x, "... d_model, ... d_model -> ...")
        # rms 形状: (..., 1) 需要 keepdim 以便广播
        rms = torch.sqrt(square_sum / self.d_model + self.eps).unsqueeze(-1)
        x = x / rms
        result = einsum(x, g, "... d_model, d_model -> ... d_model")

        return result.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def get_compatible_dff(d_model: int) -> int:
    """
    Returns the nearest multiple of 64 to 8/3 * d_model.
    """
    raw = (8 * d_model) / 3
    rounded = int((raw + 32) // 64) * 64  # round to nearest multiple of 64
    return rounded

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.linear1(x)
        x1 = silu(x1)

        x3 = self.linear3(x)

        x2 = einsum(x1, x3, "... d_ff, ... d_ff -> ... d_ff")
        x2 = self.linear2(x2)

        return x2

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len).float()
        freqs = einsum(t, inv_freq, "i, j -> i j") # 存成(max_seq_len, d_k // 2)的矩阵
        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x ({x.size(-1)}) ≠ d_k ({self.d_k}).")
        
        # 获取cached矩阵
        cos_pos = self.cos_cache[token_positions]
        sin_pos = self.sin_cache[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_odd * cos_pos + x_even * sin_pos

        out = torch.empty_like(x, device=x.device)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out








