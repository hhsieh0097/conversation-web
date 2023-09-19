import os
import pickle
import logging
import numpy as np
from typing import List

from src.config import parse_args

args = parse_args()
logger = logging.getLogger(__name__)

class MyTokenizer(object):
    def __init__(
            self, 
            pad: str = '[PAD]', 
            unk: str = '[UNK]', 
            trained_cache = None, 
        ) -> None:

        tokenizer_dir = trained_cache

        self.pad_token = pad
        self.unk_token = unk

        logger.info(f'Loading tokenizer {os.path.basename(tokenizer_dir)} from cache.')

        with open(os.path.join(tokenizer_dir, 'token2idx.pkl'), 'rb') as f:
            self.token2idx = pickle.load(f)
        with open(os.path.join(tokenizer_dir, 'idx2token.pkl'), 'rb') as f:
            self.idx2token = pickle.load(f)

        self.embedding_matrix = np.load(os.path.join(tokenizer_dir, 'embedding_matrix.npy'))
       
    @property
    def pad_token_id(self) -> int:
        return self.token2idx[self.pad_token]
    
    @property
    def unk_token_id(self) -> int:
        return self.token2idx[self.unk_token]
    
    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())
        
    def encode(self, sentence: List[str], padding=None, truncation=None, max_length=None) -> List[int]:
        encoded_ids = list()

        for token in sentence.split():
            encoded_ids.append(self.token2idx.get(token, self.unk_token_id))

        if truncation:
            encoded_ids = encoded_ids[: max_length]
        
        if padding == 'max_length' or padding == True:
            if max_length:
                if len(encoded_ids) < max_length:
                    return encoded_ids + [self.pad_token_id] * (max_length - len(encoded_ids))
                else:
                    return encoded_ids
            else:
                raise('Please set the value of max_length')

        return encoded_ids
    
    def decode(self, encoded_ids: List[int]) -> List[str]:
        sentence = list()

        for token_id in encoded_ids:
            sentence.append(self.idx2token[token_id])

        return sentence
    
    def save(self, save_dir):
        with open(os.path.join(save_dir, 'token2idx.pkl'), 'wb') as f:
            pickle.dump(self.token2idx, f)

        with open(os.path.join(save_dir, 'idx2token.pkl'), 'wb') as f:
            pickle.dump(self.idx2token, f)

        np.save(os.path.join(save_dir, 'embedding_matrix'), self.embedding_matrix)