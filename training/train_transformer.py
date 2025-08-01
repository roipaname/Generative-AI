import os
import sys
import signal
import traceback
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# Environment setup (must be before torch_xla.device())
os.environ["XLA_USE_SPMD"] = "1"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import model after appending project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.QA_transformer import QA_TransformerModel

# Hyperparameters
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 8
epochs = 10
lr = 2e-4
warmup_steps = 500
weight_decay = 0.01
max_grad_norm = 1.0
save_every_n_epochs = 3
eval_every_n_steps = 200
max_train_samples = 2000
max_eval_samples = 200

class EnhancedQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=384):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample['question']
        context = sample['context']
        answers = sample['answers']
        answer_text = answers['text'][0] if answers['text'] else ""
        answer_start = answers['answer_start'][0] if answers['answer_start'] else 0

        encoding = self.tokenizer(
            question, context,
            max_length=self.max_length,
            truncation='only_second',
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offset_mapping = encoding['offset_mapping'].squeeze(0)
        start_positions = end_positions = 0
        if answer_text:
            for i, (start, end) in enumerate(offset_mapping):
                if start <= answer_start < end:
                    start_positions = i
                if start < answer_start + len(answer_text) <= end:
                    end_positions = i
                    break
            if end_positions < start_positions:
                end_positions = start_positions

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(start_positions, dtype=torch.long),
            'end_positions': torch.tensor(end_positions, dtype=torch.long)
        }

def simple_train_fn(rank):
    try:
        device = xm.xla_device()
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        test_tensor = torch.randint(0, vocab_size, (2, max_len), dtype=torch.long).to(device)
        model(test_tensor)
        print(f"Process {rank}: simple test passed on device {device}")
    except Exception as e:
        print(f"Process {rank}: simple test failed - {e}")
        traceback.print_exc()

def full_train_fn(rank):
    try:
        device = xm.xla_device()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_prefix_space=True)
        train_data = load_dataset("squad", split=f"train[:{max_train_samples}]")
        eval_data = load_dataset("squad", split=f"validation[:{max_eval_samples}]")

        train_dataset = EnhancedQADataset(train_data, tokenizer, max_len)
        eval_dataset = EnhancedQADataset(eval_data, tokenizer, max_len)

        world_size = xm.xrt_world_size()
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)

        per_core_batch_size = max(1, batch_size // world_size)
        train_loader = DataLoader(train_dataset, batch_size=per_core_batch_size, sampler=train_sampler, drop_last=True)
        eval_loader = DataLoader(eval_dataset, batch_size=per_core_batch_size, sampler=eval_sampler)

        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, len(train_loader)*epochs)
        criterion = nn.CrossEntropyLoss()

        global_step = 0
        for epoch in range(epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                optimizer.zero_grad()
                start_logits, end_logits = model(input_ids, mask=attention_mask)
                loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                xm.optimizer_step(optimizer)
                scheduler.step()
                global_step += 1

                if global_step % eval_every_n_steps == 0 and xm.is_master_ordinal():
                    print(f"[Rank {rank}] Step {global_step} - Loss: {loss.item():.4f}")

        if xm.is_master_ordinal():
            print("Training complete.")

    except Exception as e:
        print(f"[Rank {rank}] Exception: {e}")
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    print("\n=== Testing TPU connectivity and functionality ===")
    xmp.spawn(simple_train_fn, args=(), nprocs=8, start_method='fork')


    print("\n=== Starting full TPU training ===")
    start_time = datetime.now()
    xmp.spawn(full_train_fn, args=(), nprocs=8, start_method='fork')
    print(f"Training completed in {datetime.now() - start_time}")