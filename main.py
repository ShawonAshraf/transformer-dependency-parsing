import os
import argparse
import torch
import pytorch_lightning as pl

from data.dataset import Conll06Dataset
from torch.utils.data import DataLoader
from model.parser import ParserTransformer

from data.preprocess import preprocess

argparser = argparse.ArgumentParser()

# language: en | de
argparser.add_argument("--lang", required=True, type=str)
# path to train dataset
argparser.add_argument("--train", required=True, type=str)
# path to dev dataset
argparser.add_argument("--dev", required=True, type=str)
# path to preprocessed vocab and rel file
argparser.add_argument("--pre", required=True, type=str)

# number of batches
argparser.add_argument("--batches", required=True, type=int)
# number of cpu workers
argparser.add_argument("--workers", required=True, type=int, default=1)

# epochs
argparser.add_argument("--epochs", required=True, type=int)
# use half precision or not
argparser.add_argument("--precision", required=False, type=int, default=16)
# use tensor core precision
# available on newer gpus only
# medium | high | highest
argparser.add_argument("--tf32", required=False, type=str, default="high")

args = argparser.parse_args()

if __name__ == "__main__":
    # set tf32
    torch.set_float32_matmul_precision(args.tf32)

    assert os.path.exists(args.train)
    assert os.path.exists(args.dev)

    # run preprocess on the train set only
    if not os.path.exists(args.pre):
        preprocess(args.pre, args.train)

    trainset = Conll06Dataset(args.train, args.pre)
    devset = Conll06Dataset(args.dev, args.pre)

    loader_config = {
        "pin_memory": True,
        "batch_size": args.batches,
        "num_workers": args.workers
    }

    train_loader = DataLoader(trainset, **loader_config, shuffle=True)
    dev_loader = DataLoader(devset, **loader_config, shuffle=False)

    model = ParserTransformer(vocab_size=trainset.vocab_size,
                              max_len=trainset.MAX_LEN,
                              n_parser_heads=trainset.MAX_LEN,
                              n_rels=trainset.n_rels, ignore_index=trainset.rel_dict["<PAD>"])
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        gradient_clip_val=0.1,
        val_check_interval=100
    )

    trainer.fit(model, train_loader, dev_loader)
