import os
import sys
import torch
import torch_xla.core.xla_model as xm

# Adjust this import path as needed
sys.path.append('./models/transformer')
from QA_transformer import QA_TransformerModel  # Make sure this path is correct

# Hyperparameters (adjust to match your model)
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

def main():
    print("Testing TPU connectivity...")
    try:
        device = xm.xla_device()
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Failed to get TPU device: {e}")
        sys.exit(1)
    
    try:
        print("Creating model...")
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        print("Model created successfully")

        print("Running dummy forward pass...")
        # Create dummy input tensor (batch_size=2, seq_len=10, d_model)
        dummy_input = torch.randn(2, 10, d_model).to(device)

        # Forward pass (adjust if your model input is different)
        output = model(dummy_input)
        print("Forward pass output:", output)

        print("Simple TPU test passed!")

    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
