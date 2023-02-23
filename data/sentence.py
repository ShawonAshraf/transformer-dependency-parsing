from dataclasses import dataclass
from typing import List

from .conll06_token import Conll06Token


@dataclass
class Sentence:
    tokens: List[Conll06Token]

    def __str__(self) -> str:
        s = ""
        for token in self.tokens:
            s += str(token)

        return s + "\n"
