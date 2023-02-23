import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    d_model: int
    dropout_prob: float = 0.1
    max_len: int = 150

    def setup(self) -> None:
        positions = jnp.arange(0, self.max_len)
        # vectorise
        positions = rearrange(positions, "n -> n 1")

        # but in log space
        # -2i * n / d
        # even steps , since 2i
        # n = 10e3
        denominator = -jnp.arange(0, self.d_model, 2) * jnp.log(jnp.array(10.0e3)) / self.d_model
        # exp since we took log from the original equation, which was 1/n^(2i / d)
        denominator = jnp.exp(denominator)

        # positional encoding tensor
        pe = jnp.zeros(shape=(self.max_len, 1, self.d_model))
        # encode the first dim
        pe[:, 0, 0::2] = jnp.sin(positions * denominator)
        # second dim
        pe[:, 0, 1::2] = jnp.cos(positions * denominator)

        # similar to torch buffer
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html#Positional-encoding
        self.pe = jax.device_put(pe)

        # dropout
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)
