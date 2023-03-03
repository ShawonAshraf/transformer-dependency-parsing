from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .io import read_conll06_file
from .sentence import Sentence
from .preprocess import load


class Conll06Dataset(Dataset):

    # ============== dataset methods =============

    """
        file_path: path to the dataset file
        preprocessed_info_path: path of the preprocessed json file
    """

    def __init__(self, file_path: str, preprocessed_info_path: str, MAX_LEN=150) -> None:
        self.file_path = file_path
        self.MAX_LEN = MAX_LEN

        # read sentences from file
        self.sentences = read_conll06_file(self.file_path)

        # get preprocessed vocab and rels
        pre = load(preprocessed_info_path)

        self.vocab = pre["vocabulary"]
        self.vocab_size = len(self.vocab.keys())

        # for relation label
        self.rel_dict = pre["rel_labels"]
        self.n_rels = len(list(self.rel_dict.keys()))

        # for head
        self.PAD_IDX_FOR_HEAD = self.MAX_LEN - 1

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

    # encode sentence
    def __encode_one_sentence(self, sentence: Sentence) -> torch.Tensor:
        # encode and pad basically
        # fill with pad tokens by default
        encoded = torch.ones(self.MAX_LEN) * self.vocab["<PAD>"]
        # index 0 is always the ROOT
        encoded[0] = self.vocab["<ROOT>"]

        tokens = sentence.tokens
        for idx, token in enumerate(tokens):
            if token.form in self.vocab.keys():
                encoded[idx + 1] = self.vocab[token.form]
            else:
                encoded[idx + 1] = self.vocab["<OOV>"]

        return encoded

    # encode labels and get head
    # encode and pad basically
    def __encode_rel_and_get_head(self, sentence: Sentence) -> Tuple[torch.Tensor, torch.Tensor]:

        heads = torch.ones(self.MAX_LEN, dtype=torch.float) * \
            self.PAD_IDX_FOR_HEAD

        rels = torch.ones(self.n_rels, dtype=torch.float) * \
            self.rel_dict["<PAD>"]

        for _, token in enumerate(sentence.tokens):
            heads[token.head] = token.head
            rel_idx = self.rel_dict[token.rel]
            rels[rel_idx] = rel_idx

        return heads, rels
