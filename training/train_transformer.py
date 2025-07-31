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
batch_size = 32  # Reduced batch size per core
epochs = 3
lr = 3e-4

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

        # Initialize tokenizer inside the dataset to avoid multiprocessing issues
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
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
            'start_positions': torch.tensor(start_pos, dtype=torch.long),
            'end_positions': torch.tensor(end_pos, dtype=torch.long)
        }

def train_fn(rank, flags):
    # Get proper TPU device and world size
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    
    print(f"Running on device: {device}, rank: {rank}, world_size: {world_size}")

    # Load dataset
    dataset = load_dataset("squad")
    train_dataset = QADataset(dataset["train"])
    
    # Create distributed sampler with correct world size
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        drop_last=True,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    # Wrap with TPU parallel loader
    para_loader = pl.MpDeviceLoader(train_loader, device)

    # Model, optimizer, loss
    model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize wandb only in main process
    if xm.is_master_ordinal():
        wandb.init(project="transformer-qa", name="qa-training-tpu")

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for i, batch in enumerate(para_loader):
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

            if i % 50 == 0 and xm.is_master_ordinal():
                print(f"Epoch {epoch+1} Step {i} Loss: {loss.item():.4f}")
                if 'wandb' in globals():
                    wandb.log({"loss": loss.item(), "epoch": epoch+1})

        # Print epoch completion
        if xm.is_master_ordinal():
            print(f"Completed epoch {epoch+1}")

    # Save checkpoint only on master process
    if xm.is_master_ordinal():
        os.makedirs("checkpoints", exist_ok=True)
        xm.save(model.state_dict(), "checkpoints/qa_transformer_tpu.pt")
        print("Saved QA model checkpoint.")

if __name__ == "__main__":
    # Use spawn instead of fork to avoid crashes
    xmp.spawn(train_fn, args=(dict(),), nprocs=None, start_method='spawn')