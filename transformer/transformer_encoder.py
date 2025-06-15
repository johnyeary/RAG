import torch
import torch.nn as nn
from transformer.encoder_layer import EncoderLayer
from transformer.embeddings import TokenPositionalEmbedding
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 64,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        # Token & positional embeddings
        self.embedding = TokenPositionalEmbedding(vocab_size = vocab_size, d_model = d_model, max_len = max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,num_heads,d_ff,dropout)
            for _ in range(num_layers)
        ])

    def forward(self, token_ids, mask=None):
        x = self.embedding(token_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x,mask)
        
        return x