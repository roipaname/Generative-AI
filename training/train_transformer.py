import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.transformer import TransformerModel
from models.transformer.QA_transformer import QA_TransformerModel

# Hyperparameters
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 128
epochs = 3
lr = 3e-4

# Initialize tokenizer and dataset (load once)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("squad")

class QADataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample['question']
        context = sample['context']
        answers = sample['answers']

        encoding = tokenizer(
            question,
            context,
            max_length=max_len,
            truncation='only_second',
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offsets = encoding['offset_mapping'].squeeze(0)

        answer_text = answers['text'][0] if len(answers['text']) > 0 else ''
        answer_start_char = answers['answer_start'][0] if len(answers['answer_start']) > 0 else 0

        start_pos = 0
        end_pos = 0

        if answer_text:
            for i, (start, end) in enumerate(offsets.tolist()):
                if start <= answer_start_char < end:
                    start_pos = i
                if start < answer_start_char + len(answer_text) <= end:
                    end_pos = i
                    break

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_pos,
            'end_positions': end_pos
        }

def train_fn(rank, flags):
    # rank: TPU core id (0-7 for 8 TPU cores)
    device = xm.xla_device()

    # Create dataset and DataLoader with distributed sampler
    train_dataset = QADataset(dataset["train"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=8,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size // 8, sampler=train_sampler, drop_last=True)

    # Model, optimizer, loss
    model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize wandb only in main process
    if rank == 0:
        wandb.init(project="transformer-qa", name="qa-training-tpu")

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids, mask=attention_mask)
            loss_start = criterion(start_logits, start_positions)
            loss_end = criterion(end_logits, end_positions)
            loss = (loss_start + loss_end) / 2

            loss.backward()
            xm.optimizer_step(optimizer)  # TPU-specific optimizer step

            if i % 50 == 0 and rank == 0:
                print(f"Epoch {epoch+1} Step {i} Loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(), "epoch": epoch+1})

    # Save checkpoint only on rank 0 process
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        xm.save(model.state_dict(), "checkpoints/qa_transformer_tpu.pt")
        print("Saved QA model checkpoint.")

if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(train_fn, args=(), nprocs=8, start_method='fork')

