import torch
import torch.nn as nn
import torch.nn.Functional as F
from transformer.attention import MultiHeadSelfAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, num_heads:int,d_ff:int,dropout:float = 1.0):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # Feed forward sublayer
        self.ff = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff,d_model)
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        #self attention sublayer
        attn_out, _ = self.self_attn(x,mask)
        attn_out = self.attn_dropout(attn_out)
        x = self.attn_norm(x+attn_out)  #residual + norm

        #Feed forward sublayer
        ff_out = self.ff(x)
        ff_out = self.ff_dropout(ff_out)
        out = self.ff_norm(x+ff_out) #residual + norm

        return out