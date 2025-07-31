import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# Test basic TPU connectivity first
print("Testing TPU connectivity...")
try:
    device_count = xm.xrt_world_size()
    print(f"TPU devices available: {device_count}")
    device = xm.xla_device()
    print(f"Current device: {device}")
except Exception as e:
    print(f"TPU connection error: {e}")
    sys.exit(1)

# Test imports
print("Testing model imports...")
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.transformer.QA_transformer import QA_TransformerModel
    print("Model import successful")
except Exception as e:
    print(f"Model import error: {e}")
    sys.exit(1)

# Hyperparameters
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 32  # Reduced for debugging
epochs = 1  # Reduced for debugging
lr = 3e-4

def simple_train_fn(rank):
    """Simplified training function for debugging"""
    print(f"Process {rank} starting...")
    
    try:
        device = xm.xla_device()
        print(f"Process {rank} using device: {device}")
        
        # Test model creation
        print(f"Process {rank}: Creating model...")
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        print(f"Process {rank}: Model created successfully")
        
        # Test basic tensor operations
        print(f"Process {rank}: Testing tensor operations...")
        test_tensor = torch.randn(2, 10, d_model).to(device)
        output = model(test_tensor)

        print(f"Process {rank}: Tensor operations successful")
        
        print(f"Process {rank}: Completed successfully")
        
    except Exception as e:
        print(f"Process {rank} error: {e}")
        import traceback
        traceback.print_exc()

def full_train_fn(rank):
    """Full training function"""
    print(f"Process {rank} starting full training...")
    
    try:
        device = xm.xla_device()
        
        # Initialize tokenizer and dataset
        print(f"Process {rank}: Loading tokenizer and dataset...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load a small subset for testing
        dataset = load_dataset("squad", split="train[:100]")  # Only 100 samples for testing
        
        class SimpleQADataset(torch.utils.data.Dataset):
            def __init__(self, dataset, tokenizer):
                self.dataset = dataset
                self.tokenizer = tokenizer
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                sample = self.dataset[idx]
                question = sample['question']
                context = sample['context']
                
                encoding = self.tokenizer(
                    question,
                    context,
                    max_length=max_len,
                    truncation='only_second',
                    padding='max_length',
                    return_tensors="pt"
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'start_positions': torch.tensor(0, dtype=torch.long),  # Dummy for testing
                    'end_positions': torch.tensor(1, dtype=torch.long)     # Dummy for testing
                }
        
        train_dataset = SimpleQADataset(dataset, tokenizer)
        
        world_size = xm.xrt_world_size()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        per_core_batch_size = max(1, batch_size // world_size)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=per_core_batch_size, 
            sampler=train_sampler, 
            drop_last=True,
            num_workers=0
        )
        
        print(f"Process {rank}: Dataset loaded, {len(train_dataset)} samples")
        
        # Model, optimizer, loss
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Process {rank}: Starting training...")
        
        for epoch in range(epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            
            for i, batch in enumerate(train_loader):
                if i >= 3:  # Only process 3 batches for testing
                    break
                    
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
                xm.optimizer_step(optimizer)
                
                if xm.is_master_ordinal():
                    print(f"Process {rank} Epoch {epoch+1} Step {i} Loss: {loss.item():.4f}")
            
            xm.mark_step()
        
        if xm.is_master_ordinal():
            print("Training completed successfully!")
            
    except Exception as e:
        print(f"Process {rank} full training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting TPU training debug...")
    
    # First test simple functionality
    print("\n=== Testing simple TPU functionality ===")
    try:
        xmp.spawn(simple_train_fn, args=(), nprocs=1, start_method='spawn')  # Test with 1 process first
        print("Simple test passed!")
    except Exception as e:
        print(f"Simple test failed: {e}")
        sys.exit(1)
    
    # Then test full training
    print("\n=== Testing full training functionality ===")
    try:
        xmp.spawn(full_train_fn, args=(), nprocs=None, start_method='spawn')
        print("Full training test passed!")
    except Exception as e:
        print(f"Full training test failed: {e}")
        import traceback
        traceback.print_exc()