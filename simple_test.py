# simple_test.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Set path to import custom model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/transformer')))
from models.transformer.QA_transformer import QA_TransformerModel

# Config
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

def main():
    print("Testing TPU connectivity...")
    device = xm.xla_device()
    print(f"Using device: {device}")

    print("Creating model...")
    model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
    print("Model created successfully")

    print("Running dummy forward pass...")
    dummy_input = torch.randint(0, vocab_size, (2, 10), dtype=torch.long).to(device)
    output = model(dummy_input)
    print("Forward pass output:", output)

if __name__ == "__main__":
    main()
