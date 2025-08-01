import os
import sys
import time
import torch
import string
import re
from transformers import AutoTokenizer
from datasets import load_dataset

import torch_xla
import torch_xla.core.xla_model as xm

# === Environment Setup ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PJRT_DEVICE"] = "TPU"

# Add parent directory to path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.QA_transformer import QA_TransformerModel

# === Config ===
CONFIG = {
    "vocab_size": 30522,
    "max_len": 384,
    "d_model": 512,
    "num_heads": 8,
    "d_ff": 2048,
    "num_layers": 6,
    "num_samples": 200,
    "checkpoint_path": os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pt')
}

# === Answer Normalization Helpers ===
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in string.punctuation)
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
device = xm.xla_device()
model = QA_TransformerModel(
    CONFIG["vocab_size"],
    CONFIG["d_model"],
    CONFIG["num_heads"],
    CONFIG["d_ff"],
    CONFIG["num_layers"],
    CONFIG["max_len"]
)

model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location='cpu'))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === Load Dataset ===
dataset = load_dataset("squad", split=f"validation[:{CONFIG['num_samples']}]")

# === Evaluation ===
total_em, total_f1 = 0, 0
print(f"\n=== Evaluation on SQuAD (first {CONFIG['num_samples']} samples) ===\n")

start_time = time.time()
for i, item in enumerate(dataset):
    question, context = item["question"], item["context"]
    true_answers = item["answers"]["text"]
    true_answer = true_answers[0] if true_answers else ""

    # Tokenize
    encoding = tokenizer(
        question,
        context,
        max_length=CONFIG["max_len"],
        truncation='only_second',
        padding='max_length',
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask=attention_mask)
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()

    if end_idx < start_idx:
        end_idx = start_idx

    pred_tokens = input_ids[0][start_idx:end_idx + 1].cpu()
    pred_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

    # Metrics
    em = compute_exact(true_answer, pred_answer)
    f1 = compute_f1(true_answer, pred_answer)
    total_em += em
    total_f1 += f1

    if i < 5:  # Show a few sample results
        print(f"[{i+1}] Q: {question}")
        print(f"     Predicted: {pred_answer}")
        print(f"     Ground Truth: {true_answer}")
        print(f"     EM: {em}, F1: {f1:.2f}\n")

# === Final Metrics ===
count = CONFIG["num_samples"]
avg_em = total_em / count * 100
avg_f1 = total_f1 / count * 100
elapsed = time.time() - start_time

print(f"\n📊 Final Evaluation over {count} samples:")
print(f"   Exact Match (EM): {avg_em:.2f}%")
print(f"   F1 Score:         {avg_f1:.2f}%")
print(f"   Elapsed Time:     {elapsed:.2f}s")
