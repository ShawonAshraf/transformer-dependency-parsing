import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from einops import rearrange
from trax.models import Transformer

from positional_encoder import PositionalEncoding


class Parser(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    dropout_rate: float
    max_len: int
    n_rels: int
    mode: str

    def setup(self) -> None:
        # https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Embed.html
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.positional_encoder = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)

        # transfomer
        self.transformer = Transformer(
            input_vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            max_len=self.max_len,
            mode=self.mode,
            dropout=self.dropout_rate,
            d_ff=50
        )

        self.head_predictor = nn.Dense(self.d_model, self.max_len)
        self.rel_predictor = nn.Dense(self.d_model, self.n_rels)

    def __call__(self, source: jnp.ndarray, target: jnp.ndarray):
        src = self.embedding(source) * jnp.sqrt(jnp.array(self.d_model))
        tgt = self.embedding(target) * jnp.sqrt(jnp.array(self.d_model))

        # src = rearrange(src, "bs seq embed -> seq bs embed")
        # tgt = rearrange(tgt, "bs seq embed -> seq bs embed")

        src_pe = self.positional_encoder(src)
        tgt_pe = self.positional_encoder(tgt)

        out = self.transformer(src_pe, tgt_pe)
        head_logits = self.head_predictor(out)
        rel_logits = self.rel_predictor(out)

        return jax.nn.log_softmax(head_logits), jax.nn.log_softmax(rel_logits)


if __name__ == "__main__":
    p = Parser(vocab_size=50,
               d_model=256,
               n_heads=2,
               n_encoder_layers=2,
               n_decoder_layers=2,
               dropout_rate=0.1,
               max_len=50,
               n_rels=8,
               mode="train")

    rng = jax.random.PRNGKey(0)
    rng, subkey = jax.random.split(rng)

    src = jnp.arange(50)
    tgt = jnp.arange(50)

    # src = rearrange(jnp.arange(50), "n -> 1 n")
    # tgt = rearrange(jnp.arange(50), "n -> 1 n")

    params = p.init(rng, src, tgt)
    print(params)

