import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.transformer.transformer import TransformerModel
from datasets import load_dataset
from tokenizers import Tokenizer
import wandb

# Hyperparameters
vocab_size = 30522
max_len = 128
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 32
epochs = 3
lr = 3e-4

# Load tokenizer and dataset (e.g., SQuAD)
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("squad")

# Dummy processing
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.data = [tokenizer.encode(text["context"]).ids[:max_len] for text in texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        x = torch.tensor(ids, dtype=torch.long)
        return x, x  # input == target for LM

train_data = SimpleDataset(dataset["train"])
loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

# Model, loss, optimizer
model = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

wandb.init(project="transformer-scratch")

for epoch in range(epochs):
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
            wandb.log({"loss": loss.item()})
