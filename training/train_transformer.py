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
import torch_xla.distributed.parallel_loader as pl

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

def create_qa_dataset():
    """Create dataset outside of multiprocessing to avoid issues"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("squad")
    
    class QADataset(torch.utils.data.Dataset):
        def __init__(self, dataset, tokenizer):
            self.dataset = dataset
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            sample = self.dataset[idx]
            question = sample['question']
            context = sample['context']
            answers = sample['answers']

            encoding = self.tokenizer(
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
                'start_positions': torch.tensor(start_pos, dtype=torch.long),
                'end_positions': torch.tensor(end_pos, dtype=torch.long)
            }
    
    return QADataset(dataset["train"], tokenizer)

def train_fn(rank, flags):
    # rank: TPU core id (0-7 for 8 TPU cores)
    device = xm.xla_device()
    
    print(f"Process {rank} starting on device {device}")

    # Create dataset and DataLoader
    train_dataset = create_qa_dataset()
    
    # For TPU, we need to use the world size (number of TPU cores)
    world_size = xm.xrt_world_size()
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Adjust batch size per core
    per_core_batch_size = batch_size // world_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=per_core_batch_size, 
        sampler=train_sampler, 
        drop_last=True,
        num_workers=0  # Important: set to 0 for TPU
    )

    # Use ParallelLoader for better TPU performance
    train_loader = pl.ParallelLoader(train_loader, [device])

    # Model, optimizer, loss
    model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize wandb only in main process (rank 0)
    if xm.is_master_ordinal():
        wandb.init(project="transformer-qa", name="qa-training-tpu")

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for i, batch in enumerate(train_loader.per_device_loader(device)):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']

            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids, mask=attention_mask)
            
            loss_start = criterion(start_logits, start_positions)
            loss_end = criterion(end_logits, end_positions)
            loss = (loss_start + loss_end) / 2

            loss.backward()
            xm.optimizer_step(optimizer)  # TPU-specific optimizer step

            # Log and print only from master process
            if i % 50 == 0 and xm.is_master_ordinal():
                loss_item = loss.item()
                print(f"Epoch {epoch+1} Step {i} Loss: {loss_item:.4f}")
                wandb.log({"loss": loss_item, "epoch": epoch+1, "step": i})

        # Synchronize all processes at end of epoch
        xm.mark_step()

    # Save checkpoint only on master process
    if xm.is_master_ordinal():
        os.makedirs("checkpoints", exist_ok=True)
        # Use xm.save for TPU-compatible saving
        xm.save(model.state_dict(), "checkpoints/qa_transformer_tpu.pt")
        print("Saved QA model checkpoint.")
        if 'wandb' in globals():
            wandb.finish()

if __name__ == "__main__":
    # For TPU VM, use spawn with proper process count
    # nprocs=None will automatically detect the number of TPU cores
    xmp.spawn(train_fn, args=(), nprocs=None, start_method='spawn')