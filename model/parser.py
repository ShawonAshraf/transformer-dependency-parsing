import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from .mlp import MLP


class ParserTransformer(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
