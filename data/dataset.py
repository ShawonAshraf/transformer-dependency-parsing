from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .io import read_conll06_file
from .sentence import Sentence


class Conll06Dataset(Dataset):

    # ============== dataset methods =============

    def __init__(self, file_path: str, MAX_LEN=150) -> None:
        self.file_path = file_path
        self.MAX_LEN = MAX_LEN

        # read sentences from file
        self.sentences = read_conll06_file(self.file_path)

        self.vocab = self.__build_vocab()
        self.vocab_size = len(self.vocab.keys())

        # for label
        self.rel_dict = {}
        self.__build_rel_dict()
        self.pad_idx = self.rel_dict["<PAD>"]
        self.n_rels = len(list(self.rel_dict.keys()))

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # endcode sentences
        encoded = self.__encode_one_sentence(sentence)
        # encode the rels and get the heads
        heads, rels = self.__encode_rel_and_get_head(sentence)

        return {
            "sentence": encoded,
            "heads": heads,
            "rels": rels
        }

    # ============ preprocessing methods ===========

    # build vocab
    def __build_vocab(self) -> Dict:
        words = set()
        vocab = dict()

        words.add("<ROOT>")
        words.add("<PAD>")
        words.add("<OOV>")

        for sentence in self.sentences:
            tokens = sentence.tokens
            for token in tokens:
                words.add(token.form)

        for idx, w in enumerate(words):
            vocab[w] = idx

        return vocab

    # encode sentence
    def __encode_one_sentence(self, sentence: Sentence) -> torch.Tensor:
        # encode and pad basically
        # fill with pad tokens by default
        encoded = torch.ones(size=(self.MAX_LEN, )) * self.vocab["<PAD>"]
        # index 0 is always the ROOT
        encoded[0] = self.vocab["<ROOT>"]

        tokens = sentence.tokens
        for idx, token in enumerate(tokens):
            if token.form in self.vocab.keys():
                encoded[idx + 1] = self.vocab[token.form]
            else:
                encoded[idx + 1] = self.vocab["<OOV>"]

        return encoded

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
        heads = torch.zeros(self.MAX_LEN, dtype=torch.float)
        rels = torch.zeros(self.n_rels, dtype=torch.float)

        for _, token in enumerate(sentence.tokens):
            heads[token.head] = token.head
            rel_idx = self.rel_dict[token.rel]
            rels[rel_idx] = rel_idx

        return heads, rels
