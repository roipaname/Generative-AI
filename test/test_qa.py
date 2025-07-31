import os
import sys
import torch
from transformers import AutoTokenizer

# Add models directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'transformer')))
from models.transformer.QA_transformer import QA_transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (match training)
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load model and weights
model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
checkpoint_path = os.path.join("checkpoints", "qa_transformer.pt")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def answer_question(question, context):
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

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask=attention_mask)

    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    if start_idx > end_idx:
        return "Unable to find answer."

    tokens = input_ids[0][start_idx:end_idx + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)

    return answer

if __name__ == "__main__":
    # Example question and context
    question = "What is the capital of France?"
    context = "France is a country in Europe. Paris is its capital and largest city."

    print(f"Question: {question}")
    print(f"Answer: {answer_question(question, context)}")
