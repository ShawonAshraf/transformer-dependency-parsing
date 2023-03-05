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
                 ignore_index: int,
                 mlp_hidden: int = 300,
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
        self.sentences_head_transformer = None
        self.relu1 = nn.ReLU()

        self.sentence_rel_transformer = None
        self.relu2 = nn.ReLU()

        # mlp for classification
        self.head_classifier = MLP(
            self.d_model, self.mlp_hidden, self.n_parser_heads)
        self.rel_classifier = MLP(self.d_model, self.mlp_hidden, self.n_rels)

        # loss function
        self.criterion = nn.NLLLoss(ignore_index=ignore_index)

    def forward(self, sentence: torch.Tensor,
                heads: torch.Tensor,
                rels: torch.Tensor):

        s = self.embedding(sentence) * torch.sqrt(torch.tensor(self.d_model))
        h = self.embedding(heads) * torch.sqrt(torch.tensor(self.d_model))
        r = self.embedding(rels) * torch.sqrt(torch.tensor(self.d_model))

        s = rearrange(s, "bs seq embed -> seq bs embed")
        h = rearrange(h, "bs seq embed -> seq bs embed")
        r = rearrange(r, "bs seq embed -> seq bs embed")

        s = self.positional_encoder(s)
        h = self.positional_encoder(h)
        r = self.positional_encoder(r)

        out1 = self.sentences_head_transformer(s, h)
        out1 = self.relu1(out1)
        out1 = self.head_classifier(out1)

        out2 = self.sentence_rel_transformer(s, r)
        out2 = self.relu2(out2)
        out2 = self.rel_classifier(out2)

        return F.log_softmax(out1, dim=-1), F.log_softmax(out2, dim=-1)

    def configure_optimizers(self):
        return optim.AdamW(params=self.parameters())

    def training_step(self, batch, batch_idx):
        sentence = batch["sentence"]
        heads = batch["heads"]
        rels = batch["rels"]

        out1, out2 = self(sentence, heads, rels)

        # rearrange for loss
        out1 = rearrange(out1, "seq bs probas -> bs probas seq")
        out2 = rearrange(out2, "seq bs probas -> bs probas seq")

        loss = self.criterion(out1, heads) + self.criterion(out2, rels)

        return {
            "loss": loss,
            "log": {
                "training_loss": loss
            }
        }

    def validation_step(self, batch, batch_idx):
        sentence = batch["sentence"]
        heads = batch["heads"]
        rels = batch["rels"]

        out1, out2 = self(sentence, heads, rels)

        # rearrange for loss
        out1 = rearrange(out1, "seq bs probas -> bs probas seq")
        out2 = rearrange(out2, "seq bs probas -> bs probas seq")

        loss = self.criterion(out1, heads) + self.criterion(out2, rels)

        self.log("validation_loss", loss, prog_bar=True)

        return {
            "validation_ loss": loss,
            "log": {
                "validation_loss": loss
            }
        }
