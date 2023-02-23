import jax
import jax.numpy as jnp
import flax
import flax.linen as nn


class PositionalEncoding(nn.Module):
    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        pass


class ParserTransformer(nn.Module):
    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        pass
