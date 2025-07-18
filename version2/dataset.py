import torch
from torch.utils.data import Dataset
import os

class CharDataset(Dataset):
    def __init__(self, filepath=None, block_size=128, use_preprocessed=True):
        self.block_size = block_size
        self.use_preprocessed = use_preprocessed
        if use_preprocessed and all(os.path.exists(f) for f in ['char_data.pt', 'stoi.pt', 'itos.pt']):
            # Load preprocessed data
            self.data = torch.load('char_data.pt')
            self.stoi = torch.load('stoi.pt')
            self.itos = torch.load('itos.pt')
            self.chars = sorted(self.stoi.keys(), key=lambda x: self.stoi[x])
            self.vocab_size = len(self.chars)
        elif filepath is not None:
            # Fallback: process raw text
            with open(filepath, 'r', encoding='utf-8') as f:
                self.text = f.read()
            self.chars = sorted(list(set(self.text)))
            self.vocab_size = len(self.chars)
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
            self.data = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)
        else:
            raise ValueError('Either preprocessed .pt files must exist or a filepath must be provided.')

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

    def get_vocab(self):
        return self.chars, self.stoi, self.itos

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l]) 