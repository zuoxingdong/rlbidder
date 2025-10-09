from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlbidder.constants import NUM_TICKS
from rlbidder.models.networks import init_trunc_normal


class MultiheadAttentionSDPA(nn.Module):
    """
    Multihead Attention module using Scaled Dot-Product Attention (FlashAttention 2).

    This module implements multihead attention using PyTorch's scaled_dot_product_attention
    function. 

    Args:
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
        bias (bool, optional): Whether to include bias in linear layers. Defaults to False.

    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0," \
                f" got {embed_dim=} and {num_heads=} instead"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, "
                f"got {embed_dim=} and {num_heads=}"
            )
        
        # apply QK bias
        # https://spaces.ac.cn/archives/9577
        self.qk_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # NOTE: apply QK norm 
        # see - https://arxiv.org/pdf/2409.02060#page=11.72 - Table 10
        # - https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py#L394-L414
        self.q_norm = nn.RMSNorm(embed_dim, eps=None)
        self.k_norm = nn.RMSNorm(embed_dim, eps=None)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # (batch_size, num_tokens, embed_dim)
        B, T, D = x.shape

        q, k = self.qk_proj(x).chunk(2, dim=-1)
        v = self.v_proj(x)
        
        # apply QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Split heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Combine heads and project output
        attn_output = attn_output.transpose(1, 2).view(B, T, D)
        attn_output = self.out_proj(attn_output)

        return attn_output


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshapes `freqs_cis` for broadcasting with `x`.

    Args:
        freqs_cis (torch.Tensor): A tensor of shape (seqlen, head_dim) containing complex frequency values.
        x (torch.Tensor): A tensor of shape (bs, seqlen, nheads, head_dim) representing input data.

    Returns:
        torch.Tensor: A reshaped tensor of `freqs_cis` with shape (1, seqlen, 1, head_dim) for broadcasting.
    """

    if freqs_cis.ndim == 2:  # standard RoPE
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"mismatch {freqs_cis.shape=}, {x.shape=}"
        freqs_cis = freqs_cis[None, :, None, :]
    elif freqs_cis.ndim == 3:  # batchwise, e.g. for temporal delta
        assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1]), f"mismatch {freqs_cis.shape=}, {x.shape=}"
        freqs_cis = freqs_cis[:, :, None, :]
    else:
        raise ValueError(f"Invalid freqs_cis shape: {freqs_cis.shape}")
    
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MultiheadAttentionSDPAWithRoPE(MultiheadAttentionSDPA):
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        # (batch_size, num_tokens, embed_dim)
        B, T, D = x.shape

        q, k = self.qk_proj(x).chunk(2, dim=-1)
        v = self.v_proj(x)
        
        # apply QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Split heads
        q = q.view(B, T, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        k = k.view(B, T, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Combine heads and project output
        attn_output = attn_output.transpose(1, 2).view(B, T, D)
        attn_output = self.out_proj(attn_output)

        return attn_output


# adapted from Gemma
class GatedMLP(nn.Module):
    """
    This module implements a gated MLP used in many transformer architectures.
    It consists of two projection layers, and one (gate) is followed by a SiLU activation
    and element-wise multiplication, then a final down projection.

    Args:
        hidden_size (int): The size of the input and output hidden states.
        intermediate_size (int): The size of the intermediate (expanded) representation.

    Attributes:
        gate_proj (nn.Linear): Linear layer for the gate projection.
        up_proj (nn.Linear): Linear layer for the up projection.
        down_proj (nn.Linear): Linear layer for the down projection.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        intermediate_size: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        
        self.input_layernorm = nn.RMSNorm(embedding_dim)

        self.attention = MultiheadAttentionSDPAWithRoPE(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            bias=False,
        )
        self.post_attention_layernorm = nn.RMSNorm(embedding_dim)
        
        self.mlp = GatedMLP(
            hidden_size=embedding_dim,
            intermediate_size=intermediate_size,
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.attention(self.input_layernorm(x), freqs_cis=freqs_cis) + residual

        residual = x
        x = self.residual_dropout(self.mlp(self.post_attention_layernorm(x))) + residual

        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 20,
        episode_len: int = NUM_TICKS,
        embedding_dim: int = 512,
        intermediate_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len

        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + 1, embedding_dim)
        self.state_proj = nn.Linear(state_dim, embedding_dim)
        self.state_time_fusion = nn.Sequential(
            nn.RMSNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, embedding_dim, bias=False),
            nn.Dropout(embedding_dropout),
        )
        self.action_proj = nn.Linear(action_dim, embedding_dim)
        self.action_time_fusion = nn.Sequential(
            nn.RMSNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, embedding_dim, bias=False),
            nn.Dropout(embedding_dropout),
        )
        self.return_proj = nn.Linear(1, embedding_dim)
        self.return_time_fusion = nn.Sequential(
            nn.RMSNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, embedding_dim, bias=False),
            nn.Dropout(embedding_dropout),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_ln = nn.RMSNorm(embedding_dim)

        # Assign different freqs_cis for (r, s, a) for each position in the sequence
        freqs_cis = precompute_freqs_cis(
            dim=embedding_dim // num_heads,
            max_seq_len=self.seq_len * 3,
            theta=10000,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(lambda module: init_trunc_normal(module, std=0.02))

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_size, seq_len, _ = states.shape
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_proj(states)
        state_emb = self.state_time_fusion(torch.cat([state_emb, time_emb], dim=-1))
        act_emb = self.action_proj(actions)
        act_emb = self.action_time_fusion(torch.cat([act_emb, time_emb], dim=-1))
        returns_emb = self.return_proj(returns_to_go[..., None])
        returns_emb = self.return_time_fusion(torch.cat([returns_emb, time_emb], dim=-1))

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        out = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )

        for block in self.blocks:
            out = block(out, freqs_cis=self.freqs_cis)
        out = self.output_ln(out)

        r_out, s_out, a_out = (
            out
            .reshape(batch_size, seq_len, 3, self.embedding_dim)
            .permute(2, 0, 1, 3)
        )
        return r_out, s_out, a_out


@dataclass
class DTInferenceBuffer:
    """
    A simple deque-backed temporal buffer for Decision Transformer-style inference inputs.

    Maintains the most recent `L` steps for a fixed batch size `B`, storing
    per-step tensors:
      - states: (B, S)
      - returns-to-go (rtgs): (B,)
      - timesteps: (B,) int32
      - actions_prev: (B, A) for teacher forcing (action at t-1)

    Packing behavior:
      - Outputs are LEFT-aligned and RIGHT-padded to length `L`.
      - Mask has 1.0 for valid tokens in [:n] and 0.0 for right padding.
      - actions_prev are aligned so that the first available action (from t=1)
        lands at the correct time index (i.e., actions are placed in the last
        `len(actions)` time slots of the current window).
    """
    B: int
    L: int
    state_dim: int
    action_dim: int
    dtype: npt.DTypeLike = np.float32

    states: Deque[np.ndarray] = field(init=False)
    actions: Deque[np.ndarray] = field(init=False)      # stores a_{t-1}
    rtgs: Deque[np.ndarray] = field(init=False)
    timesteps: Deque[np.ndarray] = field(init=False)
    filled: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear the buffer contents."""
        self.states = deque(maxlen=self.L)     # each item: (B, S)
        self.actions = deque(maxlen=self.L)    # each item: (B, A), stores a_{t-1}
        self.rtgs = deque(maxlen=self.L)       # each item: (B,)
        self.timesteps = deque(maxlen=self.L)  # each item: (B,)
        self.filled = 0

    def __len__(self) -> int:
        return self.filled

    def _to_array(self, x: npt.ArrayLike, dtype: npt.DTypeLike) -> np.ndarray:
        return np.array(x, dtype=dtype, copy=True)

    def _ensure_shape(self, arr: np.ndarray, expected: tuple[int, ...], name: str) -> np.ndarray:
        """
        Ensures `arr` can be reshaped to `expected` (supports squeezing a trailing singleton).
        Raises ValueError on mismatch.
        """
        try:
            if arr.shape == expected:
                return arr
            # allow (B,1) -> (B,) for vectors
            if len(expected) == 1 and arr.ndim == 2 and arr.shape[0] == expected[0] and arr.shape[1] == 1:
                return arr.reshape(expected)
            # allow (B,) -> (B,1) not needed; keep strict for matrices
            if len(expected) == 2 and arr.shape == (expected[0], expected[1]):
                return arr
        except Exception:
            pass
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")

    def append(
        self,
        state_t: npt.ArrayLike,
        rtg_t: npt.ArrayLike,
        t_t: int | npt.ArrayLike,
        action_prev: npt.ArrayLike | None = None,
    ) -> None:
        """
        Append one temporal step for a whole batch.
          - state_t: (B, S)
          - rtg_t: (B,) or (B,1)
          - t_t: scalar int or (B,) / (B,1) int-like
          - action_prev: optional (B, A) — teacher forcing (action at t-1)
        """
        B, S, A = self.B, self.state_dim, self.action_dim

        state_arr = self._ensure_shape(self._to_array(state_t, self.dtype), (B, S), "state_t")
        rtg_arr = self._to_array(rtg_t, self.dtype)
        if rtg_arr.ndim == 2 and rtg_arr.shape == (B, 1):
            rtg_arr = rtg_arr.reshape(B,)
        rtg_arr = self._ensure_shape(rtg_arr, (B,), "rtg_t")

        if np.isscalar(t_t):
            t_arr = np.full((B,), int(t_t), dtype=np.int32)
        else:
            t_arr = self._to_array(t_t, np.int32)
            if t_arr.ndim == 2 and t_arr.shape == (B, 1):
                t_arr = t_arr.reshape(B,)
            t_arr = self._ensure_shape(t_arr, (B,), "t_t")

        self.states.append(state_arr)
        self.rtgs.append(rtg_arr)
        self.timesteps.append(t_arr)

        if action_prev is not None:
            action_arr = self._ensure_shape(self._to_array(action_prev, self.dtype), (B, A), "action_prev")
            self.actions.append(action_arr)

        self.filled = min(self.filled + 1, self.L)

    def pack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pack to fixed-length arrays with LEFT alignment and RIGHT padding:
          - states: (B, L, S)
          - actions: (B, L, A) - placed in the last `len(actions)` slots so a_{t-1} aligns with s_t
          - rtgs: (B, L)
          - timesteps: (B, L) int32
          - mask: (B, L) float32 with 1.0 for valid [:n], 0.0 for right padding
        """
        B, L, S, A = self.B, self.L, self.state_dim, self.action_dim
        n = self.filled

        s_out = np.zeros((B, L, S), dtype=self.dtype)
        a_out = np.zeros((B, L, A), dtype=self.dtype)
        r_out = np.zeros((B, L), dtype=self.dtype)
        t_out = np.zeros((B, L), dtype=np.int32)
        m_out = np.zeros((B, L), dtype=self.dtype)

        if n == 0:
            return s_out, a_out, r_out, t_out, m_out

        s = np.stack(self.states, axis=1)  # (B, n, S)
        r = np.stack(self.rtgs, axis=1)    # (B, n)
        t = np.stack(self.timesteps, axis=1)  # (B, n)

        s_out[:, :n, :] = s
        r_out[:, :n] = r
        t_out[:, :n] = t
        m_out[:, :n] = 1.0

        if len(self.actions) > 0:
            a = np.stack(self.actions, axis=1)  # (B, n_a, A); first item corresponds to a0 (for t=1)
            n_a = min(a.shape[1], max(0, n - 1))  # at most n-1 previous actions are usable
            a = a[:, -n_a:, :]  # keep the most recent n-1 prev actions
            a_out[:, :n_a, :] = a  # place at indices 0..n-2 (so a_{t-1} sits at t-1)
        return s_out, a_out, r_out, t_out, m_out
