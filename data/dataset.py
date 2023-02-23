import torch
from torch.utils.data import Dataset
from .io import read_conll06_file
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp

class Conll06Dataset(Dataset):
    
    # ============== dataset methods =============
    
    def __init__(self, file_path) -> None:
        self.file_path = file_path

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
        pass

    
    
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
    
    # TODO: encode word forms
    def __encode_word_forms(self) -> None:
        pass
    
    
