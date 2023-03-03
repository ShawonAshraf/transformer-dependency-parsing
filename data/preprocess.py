from .sentence import Sentence
from typing import List, Dict, Tuple
import os
import json

"""
    generate a vocabulary dict from sentences
    vocabulary is a word -> int mapping
"""


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


"""
    build a relation label -> int mapping from sentences
"""


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


"""
    the index of the set for words and rels can change on each run, 
    so to keep them consistent and reproducible, they must be persisted on a json file
"""


def persist(vocab: Dict, rels: Dict) -> None:
    persisted_obj = {
        "vocabulary": vocab,
        "rel_labels": rels
    }

    with open("preprocessed.json", "w", encoding="utf-8") as f:
        json_str = json.dumps(persisted_obj)
        f.write(json_str)


"""
    load the persisted json file
"""


def load(file_path) -> Dict:
    assert os.path.exists(file_path)
    obj = dict()

    with open(file_path, "r", encoding="utf-8") as f:
        json_str = f.read()
        obj = json.loads(json_str)

    return obj


"""
    builds dicts and persists
"""


def preprocess(sentences: List[Sentence]):
    vocab = build_vocabulary(sentences)
    rels = build_rel_dict(sentences)

    persist(vocab, rels)
