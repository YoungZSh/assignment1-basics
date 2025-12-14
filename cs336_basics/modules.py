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
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
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
    def __init__(
        self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device).float()
        freqs = einsum(t, inv_freq, "i, j -> i j")  # 存成(max_seq_len, d_k // 2)的矩阵
        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x ({x.size(-1)}) ≠ d_k ({self.d_k}).")

        # 获取cached矩阵
        cos_pos = self.cos_cache[token_positions]  # pyright: ignore[reportIndexIssue]
        sin_pos = self.sin_cache[token_positions]  # pyright: ignore[reportIndexIssue]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_odd * cos_pos + x_even * sin_pos

        out = torch.empty_like(x, device=x.device)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values  # 转化为负数防止数值上溢，减去最大值后与原softmax等价
    x_exp = torch.exp(x)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
):
    d_k = key.size(-1)
    device, dtype = key.device, key.dtype
    scale = 1.0 / torch.tensor(d_k, device=device, dtype=dtype).sqrt()

    attn_score = einsum(query, key, "... q_len d_k, ... k_len d_k -> ... q_len k_len") * scale

    if mask is not None:
        attn_score = attn_score.masked_fill(~mask, -torch.inf)

    # softmax along the keys dimension
    attn_probs = softmax_stable(attn_score, dim=-1)

    output = einsum(attn_probs, value, "... q_len k_len, ... k_len d_v -> ... q_len d_v")

    return output


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.use_rope = use_rope

        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [
            Linear(d_model, d_model, **factory_kwargs) for _ in range(4)
        ]

        # causal mask， shape: (1, 1, max_seq_len, max_seq_len)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, max_seq_len, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # get multi-head q k v
        q, k, v = [
            rearrange(proj(x), "batch seq_len (head dim) -> batch head seq_len dim", head=self.num_heads)
            for proj in [self.q_proj, self.k_proj, self.v_proj]
        ]

        if self.use_rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)

        # 计算attention并融合不同head
        out = scaled_dot_product_attention(k, q, v, self.causal_mask[..., :seq_len, :seq_len])
        out = rearrange(out, "batch head seq_len dim -> batch seq_len (head dim)")
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        # layer1: rms_norm + attention_block
        self.pre_norm1 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.attn = CausalMultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            rope_theta=rope_theta,
            **factory_kwargs,
        )

        # layer2: rms_norm + ffn(SwiGLU)
        self.pre_norm2 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.pre_norm1(x), token_positions=token_positions)
        x = attn_out + x

        ffn_out = self.ffn(self.pre_norm2(x))
        x = ffn_out + x

        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = True,
        rope_theta: float = 1000.0,
        device: torch.Tensor | None = None,
        dtype: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.context_length = context_length

        factory_kwargs = {"device": device, "dtype": dtype}

        self.tok_emb = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            **factory_kwargs
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                use_rope=True,
                rope_theta=rope_theta,
                **factory_kwargs
            ) for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model=d_model, **factory_kwargs)

        self.linear = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        if s > self.context_length:
            raise ValueError(f"seq_len {s} exceeds context_length {self.context_length}")
        
        x = self.tok_emb(token_ids)
        pos = torch.arange(s, device = token_ids.device).unsqueeze(0).expand(b, s)

        for block in self.blocks:
            x = block(x, token_positions=pos)
        
        x = self.norm(x)

        logits = self.linear(x)

        return logits

