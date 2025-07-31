import torch
from models.transformer.transformer import TransformerModel
from tokenizers import Tokenizer
from datasets import load_dataset
from evaluate import load as load_metric

bleu = load_metric("bleu")
rouge = load_metric("rouge")

model = TransformerModel(vocab_size=30522)
model.load_state_dict(torch.load("checkpoints/transformer.pt", map_location=torch.device))

model.eval()
model.cuda()

# Tokenizer and data
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Simple eval loop
def generate(model, input_ids, max_len=64):
    model.eval()
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

references, predictions = [], []
for example in dataset["test"].select(range(10)):
    x = tokenizer.encode(example["article"]).ids[:128]
    input_tensor = torch.tensor([x], dtype=torch.long).cuda()
    output = generate(model, input_tensor, max_len=64)
    generated_text = tokenizer.decode(output[0].tolist())
    predictions.append(generated_text)
    references.append([example["highlights"]])

print("BLEU:", bleu.compute(predictions=predictions, references=references))
print("ROUGE:", rouge.compute(predictions=predictions, references=references))
