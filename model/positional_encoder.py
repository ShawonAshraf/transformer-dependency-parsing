import torch
import torch.nn as nn
from einops import rearrange

"""
    positional encoder for transformer
"""


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(0, max_len)
        # vectorize
        positions = rearrange(positions, "n -> n 1")

        # but in log space
        # -2i * n / d
        # even steps , since 2i
        # n = 10e3
        denominator = -torch.arange(0, d_model, 2) * \
            torch.log(torch.tensor(10.0e3) / d_model)
        # exp since we took log from the original equation, which was 1/n^(2i / d)
        denominator = torch.exp(denominator)

        # positional encoding tensor
        pe = torch.zeros(size=(max_len, 1, d_model))

        # encode the first dim
        pe[:, 0, 0::2] = torch.sin(positions * denominator)
        # second dim
        pe[:, 0, 1::2] = torch.cos(positions * denominator)

        # register as a buffer, variable but without gradient update
        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has a shape of (seq_len, batch_size, embedding_dim)
        # so you pass the embedded vectors for a sequence

        # residual connection + dropout
        x = x + self.positional_encoding[:x.size(0)]  # type: ignore
        return self.dropout(x)
