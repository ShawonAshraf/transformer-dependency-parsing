from typing import Tuple, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel


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
        # for lr scheduling
        self.automatic_optimization = False

        self.hidden = hidden
        self.lr = lr
        self.encoder_pretrained_name = encoder_pretrained_name
        self.ignore_index = ignore_index

        self.save_hyperparameters()

        self.n_parser_heads = n_parser_heads
        self.n_rels = n_rels

        # encoder gives word embeddings
        self.encoder = AutoModel.from_pretrained(self.encoder_pretrained_name)

        decoder_in = 384

        # decoders are linear classifiers
        self.head_decoder = nn.Linear(decoder_in, self.n_parser_heads)

        self.rel_decoder = nn.Linear(decoder_in, self.n_rels)

        # loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask)
        pooler = encoded.pooler_output

        out1 = self.head_decoder(pooler)
        # out1 = F.log_softmax(out1, dim=-1)

        out2 = self.rel_decoder(pooler)
        # out2 = F.log_softmax(out2, dim=-1)

        return out1, out2

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # for gradient clipping
        opt = self.optimizers()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        heads = batch["heads"]
        rels = batch["rels"]

        out1, out2 = self(input_ids, attention_mask)

        loss = self.criterion(out1, heads.float()) + \
            self.criterion(out2, rels.float())

        self.log("train_loss", loss, prog_bar=True)

        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.1,
                            gradient_clip_algorithm="norm")
        opt.step()

        sch = self.lr_schedulers()
        sch.step()

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        heads = batch["heads"]
        rels = batch["rels"]

        out1, out2 = self(input_ids, attention_mask)

        loss = self.criterion(out1, heads.float()) + \
            self.criterion(out2, rels.float())

        self.log("validation_loss", loss, prog_bar=True)

        return {
            "validation_ loss": loss,
            "log": {
                "validation_loss": loss
            }
        }
