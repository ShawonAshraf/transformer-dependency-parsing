from .sentence import Sentence
from typing import List, Dict, Tuple
import os
import json


def build_vocabulary(sentences: List[Sentence]) -> Dict:
    words = set()
    vocabulary = dict()

    # ROOT; OOV; PAD
    words.add("<ROOT>")
    words.add("<PAD>")
    words.add("<OOV>")

    # collect all the unique words
    for sentence in sentences:
        tokens = sentence.tokens
        for token in tokens:
            words.add(token.form)

    # map to int
    for idx, w in enumerate(words):
        vocabulary[w] = idx

    return vocabulary


def build_rel_dict(sentences: List[Sentence]) -> Dict:
    rel_set = set()
    rel_dict = dict()

    # add <PAD>
    rel_set.add("<PAD>")

    for sentence in sentences:
        tokens = sentence.tokens
        for token in tokens:
            rel_set.add(token.rel)

    # map to int
    for idx, rel in enumerate(rel_set):
        rel_dict[rel] = idx

    return rel_dict


def persist(vocab: Dict, rels: Dict) -> None:
    persisted_obj = {
        "vocabulary": vocab,
        "rel_labels": rels
    }

    with open("preprocessed.json", "w", encoding="utf-8") as f:
        json_str = json.dumps(persisted_obj)
        f.write(json_str)


def load() -> Dict:
    obj = dict()

    with open("preprocessed.json", "r", encoding="utf-8") as f:
        json_str = f.read()
        obj = json.loads(json_str)

    return obj


def preprocess(sentences: List[Sentence]) -> Dict:
    if os.path.exists("preprocessed.json"):
        obj = load()
    else:
        vocab = build_vocabulary(sentences)
        rels = build_rel_dict(sentences)

        persist(vocab, rels)

        obj = load()

    return obj
