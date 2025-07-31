import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb

# Fix import path for models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.transformer import TransformerModel

# Hyperparameters
vocab_size = 30522  # Should match tokenizer vocab
max_len = 128
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 32
epochs = 3
lr = 3e-4

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load dataset (SQuAD for testing)
dataset = load_dataset("squad")

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

# Prepare DataLoader
train_data = SimpleDataset(dataset["train"])
loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize model, loss, optimizer
model = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Logging
wandb.init(project="transformer-scratch", name="transformer-from-scratch")

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        logits = model(x)

        # Reshape for loss computation
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 50 == 0:
            avg_loss = total_loss / (i + 1)
            print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
            wandb.log({"loss": loss.item(), "epoch": epoch + 1})

# Save model
torch.save(model.state_dict(), "checkpoints/transformer_scratch.pt")
print("✅ Model saved to checkpoints/transformer_scratch.pt")
