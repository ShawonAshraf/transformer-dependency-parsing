import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
import torch.nn.functional as F
import torch.optim as optim


class ParserTransformer(pl.LightningModule):
    def __init__(self, lr: float,
                 parser_heads: int,
                 parser_rels: int) -> None:
        super().__init__()

        self.lr = lr
        self.save_hyperparameters()

        self.parser_heads = parser_heads
        self.parser_rels = parser_rels

        self.distil_bert = AutoModel.from_pretrained("distilbert-base-cased")

        self.head_labeler = nn.Linear(768, self.parser_heads)
        self.rel_labeler = nn.Linear(768, self.parser_rels)
        self.dropout = nn.Dropout(0.1)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor):
        out = self.distil_bert(x)

        head_logits = self.head_labeler(out)
        head_logits = F.relu(head_logits)

        rel_logits = self.rel_labeler(out)
        rel_logits = F.relu(rel_logits)

        return self.dropout(head_logits), self.dropout(rel_logits)

    def configure_optimizers(self):
        return optim.Adam(lr=self.lr, params=self.parameters())

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
