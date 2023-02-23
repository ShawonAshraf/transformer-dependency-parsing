import os
from typing import List

from tqdm.auto import tqdm

from .conll06_token import Conll06Token
from .sentence import Sentence


def read_conll06_file(file_path: str) -> List[Sentence]:
    assert os.path.exists(file_path)

    # list for all sentences
    sentences: List[Sentence] = []

    # read file
    with open(file_path, "r") as f:
        lines_in_file = f.readlines()

    # temp space for processing lines
    lines = []
    buffer = []
    for _, line in tqdm(enumerate(lines_in_file), desc="read_lines_from_file"):
        if line == "\n":
            lines.append(buffer)
            # clear buffer
            buffer = []
        else:
            buffer.append(line)

    # make sure that buffer is always empty after the loop ends
    assert len(buffer) == 0

    # process
    for idx, line in tqdm(enumerate(lines), desc="process_read_lines"):
        tokens = []
        for token_info in line:
            temp = token_info.split("\t")
            token = Conll06Token(*temp)
            tokens.append(token)

        # create sentence object and append
        sentences.append(Sentence(tokens))
    return sentences


# write back to file in conll06 format
def write_conll06_file(sentences: List[Sentence], out_file_path: str) -> None:
    with open(out_file_path, "w") as f:
        for sentence in sentences:
            f.write(str(sentence))
