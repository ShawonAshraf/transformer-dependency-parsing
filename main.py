import os
import argparse
import torch
import pytorch_lightning as pl

from data.dataset import Conll06Dataset
from torch.utils.data import DataLoader
from model.parser import ParserTransformer

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    train = "/home/shawon/Projects/parser-data/english/train/wsj_train.conll06"
    dev = "/home/shawon/Projects/parser-data/english/dev/wsj_dev.conll06.gold"

    assert os.path.exists(train)
    assert os.path.exists(dev)

    trainset = Conll06Dataset(train)
    devset = Conll06Dataset(dev)

    loader_config = {
        "pin_memory": True,
        "batch_size": 32,
        # "num_workers": 12
    }

    train_loader = DataLoader(trainset, **loader_config, shuffle=True)
    dev_loader = DataLoader(devset, **loader_config, shuffle=False)

    model = ParserTransformer(lr=2e-5,
                              parser_heads=trainset.MAX_LEN,
                              parser_rels=len(list(trainset.rel_dict.keys())),
                              ignore_idx=trainset.pad_idx)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        precision=16,
        gradient_clip_val=0.1,
        val_check_interval=300
    )

    trainer.fit(model, train_loader, dev_loader)
