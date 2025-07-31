import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 32
epochs = 3
lr = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Simple dataset wrapper
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.data = [
            tokenizer(text["context"], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")["input_ids"].squeeze(0)
            for text in texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # input == target

