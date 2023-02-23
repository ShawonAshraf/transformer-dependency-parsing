from typing import Tuple

import jax
import jax.numpy as jnp
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

        # vocabulary
        self.vocab = dict()
        self.__init_vocab()
        self.vocab_size = len(list(self.vocab.keys()))

        # for label
        self.rel_dict = {}
        self.__build_rel_dict()

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # encode sentence
        encoded_sentence = self.__encode_sentence(sentence)

        # encode the rels and get the heads
        heads, rels = self.__encode_rel_and_get_head(sentence)

        return {
            "sentence": encoded_sentence,
            "heads": heads,
            "rels": rels
        }

    # ============ preprocessing methods ===========

    # create vocabulary

    def __init_vocab(self) -> None:
        # word form -> int
        self.vocab = {}

        word_set = set()
        # add a OOV token
        word_set.add("<OOV>")

        for _, sentence in tqdm(enumerate(self.sentences), desc="init vocab"):
            token_forms = [token.form for token in sentence.tokens]
            for tf in token_forms:
                word_set.add(tf)

        # set contains the word forms
        # convert to integers
        for idx, word in tqdm(enumerate(word_set), desc="mapping word form to index"):
            if word not in self.vocab.keys():
                self.vocab[word] = idx

    # build rel dict

    def __build_rel_dict(self) -> None:
        rel_set = set()
        for _, sentence in tqdm(enumerate(self.sentences), desc="encoding relation labels"):
            rels = [token.rel for token in sentence.tokens]
            for rel in rels:
                rel_set.add(rel)

        # map to int
        for idx, rel in tqdm(enumerate(rel_set), desc="mapping rel to int"):
            if rel not in self.rel_dict.keys():
                self.rel_dict[rel] = idx

    # encode word forms of the tokens from a sentence
    # also pad them for vectorisation
    def __encode_sentence(self, sentence: Sentence) -> jnp.ndarray:
        # right padding
        # encoded will be populated from left to right
        encoded = jnp.zeros(shape=(self.MAX_LEN, ), dtype=jnp.float32)

        for idx, token in enumerate(sentence.tokens):
            if token.form in self.vocab.keys():
                encoded[idx] = self.vocab[token.form]
            else:
                encoded[idx] = self.vocab["<OOV>"]

        return encoded

    # encode labels and get head
    def __encode_rel_and_get_head(self, sentence: Sentence) -> Tuple[jnp.ndarray, jnp.ndarray]:
        heads = jnp.ones(shape=(self.MAX_LEN, ), dtype=jnp.float32) * -1.0
        rels = jnp.ones(shape=(self.MAX_LEN, ), dtype=jnp.float32) * -1.0

        for idx, token in enumerate(sentence.tokens):
            heads[idx] = token.head
            rels[idx] = self.rel_dict[token.rel]

        return heads, rels
