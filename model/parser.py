from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

from .mlp import MLP
from .positional_encoder import PositionalEncoder

"""
    with default transformer properties
    can be changed as well
"""


class ParserTransformer(pl.LightningModule):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 n_parser_heads: int,
                 n_rels: int,
                 mlp_hidden: int = 300,
                 n_decoder_layers: int = 6,
                 n_encoder_layers: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 lr: float = 1e-3) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.mlp_hidden = mlp_hidden
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoders = n_encoder_layers
        self.n_decoders = n_decoder_layers
        self.max_len = max_len
        self.lr = lr

        self.save_hyperparameters()

        self.n_parser_heads = n_parser_heads
        self.n_rels = n_rels

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.d_model)

        # positional encoder
        self.positional_encoder = PositionalEncoder(
            d_model=self.d_model, max_len=self.max_len)

        # transformers for alignment
        self.sentences_head_transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=self.n_encoders,
            num_decoder_layers=self.n_decoders
        )
        self.relu1 = nn.ReLU()

        self.sentence_rel_transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=self.n_encoders,
            num_decoder_layers=self.n_decoders
        )
        self.relu2 = nn.ReLU()

        # mlp for classification
        self.head_classifier = MLP(
            self.d_model, self.mlp_hidden, self.n_parser_heads)
        self.rel_classifier = MLP(self.d_model, self.mlp_hidden, self.n_rels)

    def forward(self, sentence: torch.Tensor,
                heads: torch.Tensor,
                rels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        s = self.embedding(sentence) * torch.sqrt(torch.tensor(self.d_model))
        h = heads * torch.sqrt(torch.tensor(self.d_model))
        r = rels * torch.sqrt(torch.tensor(self.d_model))

        s = rearrange(s, "bs seq embed -> seq bs embed")
        h = rearrange(h, "bs seq -> seq bs")
        r = rearrange(r, "bs seq -> seq bs")

        s = self.positional_encoder(s)
        h = self.positional_encoder(h)
        r = self.positional_encoder(r)

        out1 = self.sentences_head_transformer(s, h)
        out1 = self.head_classifier(out1)

        out2 = self.sentence_rel_transformer(s, r)
        out2 = self.rel_classifier(out2)

        return out1, out2
