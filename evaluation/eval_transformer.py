import os
import sys
import torch
from transformers import AutoTokenizer

import torch_xla
import torch_xla.core.xla_model as xm

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import your model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.QA_transformer import QA_TransformerModel

# Define model parameters
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

# Set up TPU
device = xm.xla_device()

# Initialize model and load checkpoint
model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'qa_transformer_checkpoint.pt')
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample question/context for testing
question = "What is the capital of France?"
context = "France is a country in Europe. The capital of France is Paris, which is known for the Eiffel Tower."

encoding = tokenizer(
    question,
    context,
    max_length=max_len,
    truncation='only_second',
    padding='max_length',
    return_tensors="pt"
)

input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

# Run inference
with torch.no_grad():
    start_logits, end_logits = model(input_ids, mask=attention_mask)
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

# Decode the predicted answer
tokens = input_ids[0][start_idx:end_idx + 1]
answer = tokenizer.decode(tokens, skip_special_tokens=True)

print(f"Q: {question}")
print(f"A: {answer}")
