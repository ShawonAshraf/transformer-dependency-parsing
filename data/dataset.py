from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .io import read_conll06_file
from .sentence import Sentence
from .preprocess import load

from transformers import AutoTokenizer


class Conll06Dataset(Dataset):

    # ============== dataset methods =============

    """
        file_path: path to the dataset file
        preprocessed_info_path: path of the preprocessed json file
    """

    def __init__(self, file_path: str, preprocessed: str, pretrained_model_name: str, MAX_LEN: int = 512) -> None:
        self.file_path = file_path
        self.MAX_LEN = MAX_LEN

        # read sentences from file
        self.sentences = read_conll06_file(self.file_path)
        self.preprocessed_dict = load(preprocessed)
        self.rel_dict = self.preprocessed_dict["rel_labels"]
        self.n_rels = len(self.rel_dict.keys())

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        sentence_text = "<ROOT>" + \
            "".join([tok.form for tok in sentence.tokens])
        # endcode sentences
        encoded = self.tokenizer.encode_plus(
            sentence_text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            add_special_tokens=True
        )
        # encode the rels and get the heads
        heads, rels = self.__encode_rel_and_get_head(sentence)

        return {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "heads": heads,
            "rels": rels
        }

    # ============ preprocessing methods ===========

    # encode labels and get head
    # encode and pad basically
    def __encode_rel_and_get_head(self, sentence: Sentence) -> Tuple[torch.Tensor, torch.Tensor]:

        heads = torch.ones(self.MAX_LEN, dtype=torch.long) * -1

        rels = torch.ones(self.MAX_LEN, dtype=torch.long) * -1

        for _, token in enumerate(sentence.tokens):
            heads[token.head] = token.head
            rel_idx = self.rel_dict[token.rel]
            rels[rel_idx] = rel_idx

        return heads, rels
