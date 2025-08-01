import os
import sys
import torch
from transformers import AutoTokenizer

import torch_xla
import torch_xla.core.xla_model as xm

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to import QA_TransformerModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.QA_transformer import QA_TransformerModel

# === Model Hyperparameters (same as training) ===
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

# === Load model to TPU ===
device = xm.xla_device()
model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pt')

# Load weights on CPU, then move to TPU
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.to(device)
model.eval()

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === Test Cases ===
test_cases = [
    {
        "question": "Who wrote the novel 'Pride and Prejudice'?",
        "context": "Jane Austen was an English novelist known primarily for her six major novels, including 'Pride and Prejudice'."
    },
    {
        "question": "What is the tallest mountain in the world?",
        "context": "Mount Everest, located in the Himalayas, is the tallest mountain in the world with a peak at 8,848 meters above sea level."
    },
    {
        "question": "When did the World War II end?",
        "context": "World War II ended in 1945 after the unconditional surrender of the Axis powers."
    },
    {
        "question": "What is the chemical symbol for water?",
        "context": "Water is a chemical compound consisting of two hydrogen atoms and one oxygen atom, with the chemical symbol H2O."
    },
    {
        "question": "Who is known as the father of computers?",
        "context": "Charles Babbage is often considered the 'father of the computer' for his work on the Analytical Engine in the 1830s."
    }
]

# === Inference Loop ===
print("\n=== QA Inference Results ===\n")
for idx, sample in enumerate(test_cases):
    question = sample["question"]
    context = sample["context"]

    # Tokenize
    encoding = tokenizer(
        question,
        context,
        max_length=max_len,
        truncation='only_second',
        padding='max_length',
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask=attention_mask)
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()

    # Extract and decode answer
    tokens = input_ids[0][start_idx:end_idx + 1].cpu()
    answer = tokenizer.decode(tokens, skip_special_tokens=True)

    # Display result
    print(f"Q{idx+1}: {question}")
    print(f"A{idx+1}: {answer}\n")
