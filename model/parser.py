from typing import Tuple, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from .mlp import MLP

"""
    with default transformer properties
    can be changed as well
"""


class ParserTransformer(pl.LightningModule):
    def __init__(self,
                 n_parser_heads: int,
                 n_rels: int,
                 ignore_index: int,
                 encoder_pretrained_name: str,
                 hidden: int = 300,
                 lr: float = 1e-3) -> None:
        super().__init__()

        self.hidden = hidden
        self.lr = lr
        self.encoder_pretrained_name = encoder_pretrained_name

        self.save_hyperparameters()

        self.n_parser_heads = n_parser_heads
        self.n_rels = n_rels

        # encoder gives word embeddings
        self.encoder = AutoModel.from_pretrained(self.encoder_pretrained_name)

        # transformers for alignment
        self.sentences_head_transformer = None
        self.relu1 = nn.ReLU()

        self.sentence_rel_transformer = None
        self.relu2 = nn.ReLU()

        # decoders as classifier
        # mlp

        decoder_in = 384

        self.head_decoder = MLP(
            decoder_in, self.hidden, self.n_parser_heads)
        self.rel_decoder = MLP(decoder_in, self.hidden, self.n_rels)

        # loss function
        self.criterion = nn.NLLLoss(ignore_index=ignore_index)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask)
        pooler = encoded.pooler_output

        out1 = self.head_decoder(pooler)
        out1 = F.log_softmax(out1, dim=-1)

        out2 = self.rel_decoder(pooler)
        out2 = F.log_softmax(out2, dim=-1)

        return out1, out2

    def configure_optimizers(self):
        return optim.AdamW(params=self.parameters())

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        heads = batch["heads"]
        rels = batch["rels"]

        out1, out2 = self(input_ids, attention_mask)

        loss = self.criterion(out1, heads) + self.criterion(out2, rels)

        return {
            "loss": loss,
            "log": {
                "training_loss": loss
            }
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        heads = batch["heads"]
        rels = batch["rels"]

        out1, out2 = self(input_ids, attention_mask)

        print(out1.size())
        print(out2.size())

        loss = self.criterion(out1, heads) + self.criterion(out2, rels)

        self.log("validation_loss", loss, prog_bar=True)

        return {
            "validation_ loss": loss,
            "log": {
                "validation_loss": loss
            }
        }
