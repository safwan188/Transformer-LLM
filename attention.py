from typing import Optional, Tuple
from torch import nn, Tensor
import torch
import torch.nn.functional as F
import math

def create_kqv_matrix(input_vector_dim: int, n_heads: int = 1) -> nn.Linear:
    return nn.Linear(input_vector_dim, 3 * input_vector_dim, bias=False)

def kqv(x: Tensor, linear: nn.Linear) -> Tuple[Tensor, Tensor, Tensor]:
    kqv_concat = linear(x)
    return kqv_concat.chunk(3, dim=-1)

def attention_scores(q: Tensor, k: Tensor) -> Tensor:
    d_k = q.size(-1)
    return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

def create_causal_mask(max_context_len: int) -> Tensor:
    return torch.tril(torch.ones(1, max_context_len, max_context_len))

def self_attention(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, dropout_rate: float = 0.1) -> Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = F.dropout(attention_weights, p=dropout_rate, training=True)
    return torch.matmul(attention_weights, v)

def multi_head_attention_layer(x: Tensor, kqv_matrix: nn.Linear, mask: Optional[Tensor], n_heads: int, dropout_rate: float = 0.1) -> Tuple[Tensor, Tensor]:
    B, N, D = x.size()
    k, q, v = kqv(x, kqv_matrix)
    
    k = k.view(B, N, n_heads, D // n_heads).transpose(1, 2)
    q = q.view(B, N, n_heads, D // n_heads).transpose(1, 2)
    v = v.view(B, N, n_heads, D // n_heads).transpose(1, 2)
    
    if mask is not None:
        mask = mask[:, :N, :N]
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = F.dropout(attention_weights, p=dropout_rate, training=True)
    
    sa = torch.matmul(attention_weights, v)
    
    return sa.transpose(1, 2).contiguous().view(B, N, D), attention_weights

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, max_context_len: int, dropout_rate: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.kqv_matrix = create_kqv_matrix(embed_dim)
        mask = create_causal_mask(max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.last_attention_weights = None

    def forward(self, x: Tensor) -> Tensor:
        sa, attention_weights = multi_head_attention_layer(x, self.kqv_matrix, self.mask, self.n_heads, self.dropout_rate)
        self.last_attention_weights = attention_weights
        return self.dropout(self.proj(sa))

    def get_last_attention_weights(self) -> Optional[Tensor]:
        return self.last_attention_weights