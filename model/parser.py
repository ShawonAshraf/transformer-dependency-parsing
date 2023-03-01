import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange


class ParserTransformer(pl.LightningModule):
    def __init__(self, lr: float,
                 parser_heads: int,
                 parser_rels: int,
                 ignore_idx: int) -> None:
        super().__init__()

        self.lr = lr
        self.save_hyperparameters()

        self.parser_heads = parser_heads
        self.parser_rels = parser_rels
        self.ignore_idx = ignore_idx

        self.distil_bert = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")

        self.head_labeler = nn.Linear(384, self.parser_heads)
        self.rel_labeler = nn.Linear(384, self.parser_rels)
        self.dropout = nn.Dropout(0.1)

        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.distil_bert(input_ids=input_ids, attention_mask=attention_mask)
        out = out.pooler_output

        head_logits = self.head_labeler(out)
        head_logits = F.relu(head_logits)

        rel_logits = self.rel_labeler(out)
        rel_logits = F.relu(rel_logits)

        return self.log_softmax(self.dropout(head_logits)), \
            self.log_softmax(self.dropout(rel_logits))

    def configure_optimizers(self):
        return optim.Adam(lr=self.lr, params=self.parameters())

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        heads = batch["heads"]
        rels = batch["rels"]

        heads = rearrange(heads, "bs 1 seq -> bs seq")
        rels = rearrange(rels, "bs 1 seq -> bs seq")

        head_logits, rel_logits = self(input_ids, attention_mask)

        loss = self.loss_fn(head_logits, heads) + self.loss_fn(rel_logits, rels)

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

        heads = rearrange(heads, "bs 1 seq -> bs seq")
        rels = rearrange(rels, "bs 1 seq -> bs seq")

        head_logits, rel_logits = self(input_ids, attention_mask)

        loss = self.loss_fn(head_logits, heads) + self.loss_fn(rel_logits, rels)

        self.log("validation_loss", loss, prog_bar=True)
