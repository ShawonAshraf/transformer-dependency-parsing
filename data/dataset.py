from typing import Tuple

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .io import read_conll06_file
from .sentence import Sentence


class Conll06Dataset(Dataset):

    # ============== dataset methods =============

    def __init__(self, file_path: str, MAX_LEN=150) -> None:
        self.file_path = file_path
        self.MAX_LEN = MAX_LEN

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")

        # read sentences from file
        self.sentences = read_conll06_file(self.file_path)

        # for label
        self.rel_dict = {}
        self.__build_rel_dict()
        self.pad_idx = self.rel_dict["<PAD>"]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        text = " ".join([token.form for token in sentence.tokens]).strip()

        # encode sentence
        encoded_sentence = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            max_length=self.MAX_LEN,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )

        # encode the rels and get the heads
        heads, rels = self.__encode_rel_and_get_head(sentence)

        return {
            "input_ids": encoded_sentence["input_ids"].flatten(),
            "attention_mask": encoded_sentence["attention_mask"].flatten(),
            "heads": heads.reshape(1, -1),
            "rels": rels.reshape(1, -1)
        }

    # ============ preprocessing methods ===========

    # build rel dict

    def __build_rel_dict(self) -> None:
        rel_set = set()
        for _, sentence in tqdm(enumerate(self.sentences), desc="encoding relation labels"):
            rels = [token.rel for token in sentence.tokens]
            for rel in rels:
                rel_set.add(rel)

        # add <PAD>
        rel_set.add("<PAD>")

        # map to int
        for idx, rel in tqdm(enumerate(rel_set), desc="mapping rel to int"):
            if rel not in self.rel_dict.keys():
                self.rel_dict[rel] = idx

    # encode labels and get head
    def __encode_rel_and_get_head(self, sentence: Sentence) -> Tuple[torch.Tensor, torch.Tensor]:
        # -1.0 is used as padding here
        heads = torch.ones((self.MAX_LEN,)) * self.pad_idx
        rels = torch.ones((self.MAX_LEN,)) * self.pad_idx

        for idx, token in enumerate(sentence.tokens):
            heads[idx] = token.head
            rels[idx] = self.rel_dict[token.rel]

        return heads, rels
