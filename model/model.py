import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from einops import rearrange

from .positional_encoder import PositionalEncoding


class Transformer(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    dropout_prob: float
    max_len: int
    n_tags: int

    def setup(self) -> None:
        # https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Embed.html
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.positional_encoder = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)
        self.dropout = nn.Dropout(self.dropout_prob)

        # transfomer
        self.transformer = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pass
