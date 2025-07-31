from .layers.mha import MultiHeadAttention
from .layers.feedforward import FeedForward
from .layers.positional_encoding import PositionalEncoding
import torch
import torch.nn as nn
import math


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_hidden=False):
       x = self.token_emb(x)
       x = self.pos_emb(x)
       for layer in self.layers:
         x = layer(x, mask)
       x = self.norm(x)
    
       if return_hidden:
         return x  # (B, L, d_model) — used for QA
       else:
        return self.output_proj(x)  # (B, L, vocab_size) — used for LM


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.output_proj(x)

