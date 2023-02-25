import os
import argparse


from data.dataset import Conll06Dataset
from data.collator import data_collator_fn
from torch.utils.data import DataLoader


if __name__ == "__main__":
    p = "/home/shawon/Projects/parser-data/english/train/wsj_train.conll06"

    trainset = Conll06Dataset(p)
    train_loader = DataLoader(trainset, batch_size=16, drop_last=True, num_workers=8, collate_fn=data_collator_fn)
    
    for td in train_loader:
        print(type(td["heads"]))
        # break
    