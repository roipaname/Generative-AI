import os
import sys
import torch
import string
import re
from transformers import AutoTokenizer
from datasets import load_dataset

import torch_xla
import torch_xla.core.xla_model as xm

# Set environment vars
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PJRT_DEVICE"] = "TPU"

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.QA_transformer import QA_TransformerModel

# === Evaluation Helpers ===

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()
    common = set(gold_tokens) & set(pred_tokens)
    if not gold_tokens or not pred_tokens:
        return int(gold_tokens == pred_tokens)
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)

# === Load Model & Tokenizer ===
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

device = xm.xla_device()
model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pt')
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === Load 200 SQuAD validation samples ===
dataset = load_dataset("squad", split="validation[:200]")

# === Evaluation Loop ===
total_em = 0
total_f1 = 0
count = 0

print("\n=== Evaluation on SQuAD Validation (200 samples) ===\n")

for item in dataset:
    question = item["question"]
    context = item["context"]
    true_answers = item["answers"]["text"]
    true_answer = true_answers[0] if true_answers else ""

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

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask=attention_mask)
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()

    if end_idx < start_idx:
        end_idx = start_idx

    pred_tokens = input_ids[0][start_idx:end_idx + 1].cpu()
    pred_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)

    em = compute_exact(true_answer, pred_answer)
    f1 = compute_f1(true_answer, pred_answer)

    total_em += em
    total_f1 += f1
    count += 1

    if count <= 5:
        print(f"Q: {question}")
        print(f"Pred: {pred_answer}")
        print(f"True: {true_answer}")
        print(f"EM: {em}, F1: {f1:.2f}\n")

# === Final Metrics ===
avg_em = total_em / count * 100
avg_f1 = total_f1 / count * 100

print(f"\n📊 Final Evaluation over {count} samples:")
print(f"Exact Match (EM): {avg_em:.2f}%")
print(f"F1 Score: {avg_f1:.2f}%")
