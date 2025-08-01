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
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr

# Critical: Environment setup BEFORE any torch_xla imports
os.environ["XLA_USE_SPMD"] = "1"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["XLA_PERSISTENT_CACHE_DEVICE"] = "1"
os.environ["XLA_CACHE_SIZE"] = "128MB"

# Import model after appending project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.QA_transformer import QA_TransformerModel

# Hyperparameters optimized for TPU v2-8
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 64  # Global batch size
epochs = 10
lr = 2e-4
warmup_steps = 500
weight_decay = 0.01
max_grad_norm = 1.0
save_every_n_epochs = 3
eval_every_n_steps = 100
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

def cleanup_resources():
    """Clean up TPU and multiprocessing resources"""
    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def train_fn(index):
    """Fixed training function with proper TPU handling"""
    try:
        # Initialize XLA device - this is critical
        device = xm.xla_device()
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        
        print(f"[Rank {rank}/{world_size}] Initialized on device: {device}")
        
        if rank == 0:
            print(f"Total TPU cores being used: {world_size}")
            print(f"Global batch size: {batch_size}")
            print(f"Per-core batch size: {batch_size // world_size}")
        
        # Quick connectivity test
        print(f"[Rank {rank}] Running connectivity test...")
        model_test = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        test_tensor = torch.randint(0, vocab_size, (2, max_len), dtype=torch.long).to(device)
        test_output = model_test(test_tensor)
        print(f"[Rank {rank}] Connectivity test passed! Output shapes: {[x.shape for x in test_output]}")
        
        # Clean up test objects immediately
        del model_test, test_tensor, test_output
        
        # Load datasets and tokenizer
        if rank == 0:
            print("Loading datasets...")
            
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_prefix_space=True)
        train_data = load_dataset("squad", split=f"train[:{max_train_samples}]")
        eval_data = load_dataset("squad", split=f"validation[:{max_eval_samples}]")

        train_dataset = EnhancedQADataset(train_data, tokenizer, max_len)
        eval_dataset = EnhancedQADataset(eval_data, tokenizer, max_len)
        
        if rank == 0:
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Eval dataset size: {len(eval_dataset)}")
            os.makedirs("checkpoints", exist_ok=True)

        # Setup distributed samplers
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        
        # Calculate per-core batch size
        per_core_batch_size = batch_size // world_size
        if per_core_batch_size < 1:
            per_core_batch_size = 1
            
        if rank == 0:
            print(f"Per-core batch size: {per_core_batch_size}")
            print(f"Effective global batch size: {per_core_batch_size * world_size}")
        
        # Create data loaders with proper settings for TPU
        train_loader = DataLoader(
            train_dataset, 
            batch_size=per_core_batch_size, 
            sampler=train_sampler, 
            drop_last=True,
            num_workers=0,  # Critical: No multiprocessing workers on TPU
            pin_memory=False,
            persistent_workers=False
        )
        
        # Use ParallelLoader for optimal TPU performance
        train_device_loader = pl.ParallelLoader(train_loader, [device])

        # Initialize model and training components
        if rank == 0:
            total_params = sum(p.numel() for p in QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).parameters())
            print(f"Initializing model with {total_params:,} parameters...")
        
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.CrossEntropyLoss()

        if rank == 0:
            print("Starting training loop...")
            print(f"Steps per epoch: {len(train_loader)}")
            print(f"Total training steps: {total_steps}")
        
        # Synchronize all processes before training
        xm.rendezvous('training_start')
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            step_count = 0
            
            if rank == 0:
                print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            # Use the device loader for optimal TPU performance
            for batch_idx, batch in enumerate(train_device_loader.per_device_loader(device)):
                try:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    start_positions = batch['start_positions']
                    end_positions = batch['end_positions']

                    optimizer.zero_grad()
                    start_logits, end_logits = model(input_ids, mask=attention_mask)
                    
                    start_loss = criterion(start_logits, start_positions)
                    end_loss = criterion(end_logits, end_positions)
                    loss = (start_loss + end_loss) / 2
                    
                    loss.backward()
                    
                    # Critical: Use XLA-specific gradient clipping and optimizer step
                    xm.reduce_gradients(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    step_count += 1
                    global_step += 1

                    # Periodic logging (reduce frequency to avoid overhead)
                    if global_step % eval_every_n_steps == 0 and rank == 0:
                        avg_loss = epoch_loss / step_count if step_count > 0 else 0
                        print(f"Step {global_step:4d} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
                
                except Exception as batch_e:
                    print(f"[Rank {rank}] Error in batch {batch_idx}: {batch_e}")
                    continue
            
            # Synchronize all processes before epoch summary
            xm.rendezvous(f'epoch_end_{epoch}')
            
            # Calculate and report epoch statistics
            if step_count > 0:
                avg_epoch_loss = epoch_loss / step_count
                
                if rank == 0:
                    print(f"Epoch {epoch + 1} completed | Steps: {step_count} | Avg Loss: {avg_epoch_loss:.4f}")
                    
                    # Save best model
                    if avg_epoch_loss < best_loss:
                        best_loss = avg_epoch_loss
                        # Only rank 0 saves the model
                        xm.save(model.state_dict(), "checkpoints/best_model.pt")
                        print(f"New best model saved: checkpoints/best_model.pt (Loss: {avg_epoch_loss:.4f})")
                    
                    # Periodic checkpoint
                    if (epoch + 1) % save_every_n_epochs == 0:
                        ckpt_path = f"checkpoints/model_epoch{epoch + 1}.pt"
                        xm.save(model.state_dict(), ckpt_path)
                        print(f"Checkpoint saved: {ckpt_path}")
                        
                    # Force print buffer flush
                    sys.stdout.flush()
            else:
                if rank == 0:
                    print(f"WARNING: Epoch {epoch + 1} had no training steps!")

        # Save final model
        if rank == 0:
            final_ckpt_path = "checkpoints/final_transformer_model.pt"
            xm.save(model.state_dict(), final_ckpt_path)
            print(f"Final model saved: {final_ckpt_path}")
            print(f"\nTraining completed successfully!")
            print(f"Best loss achieved: {best_loss:.4f}")
            print(f"Total cores used: {world_size}")

    except Exception as e:
        print(f"[Rank {xm.get_ordinal() if 'xm' in locals() else 'Unknown'}] Training failed: {e}")
        traceback.print_exc()
        cleanup_resources()
        raise

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    print(f"Received signal {signum}. Cleaning up...")
    cleanup_resources()
    sys.exit(1)

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== TPU v2-8 Training Configuration ===")
    print(f"TPU Type: v2-8")
    print(f"Expected TPU Cores: 8")
    print(f"Global batch size: {batch_size}")
    print(f"Per-core batch size: {batch_size // 8}")
    print(f"Effective global batch: {batch_size}")
    
    print(f"\n=== Starting TPU v2-8 Training ===")
    start_time = datetime.now()
    
    try:
        # CRITICAL FIX: Use nprocs=8 for TPU v2-8, not nprocs=1
        xmp.spawn(train_fn, nprocs=8, start_method='fork')

        print(f"\nTraining completed successfully in {datetime.now() - start_time}")
        print("Utilized all 8 TPU v2-8 cores")
        
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        cleanup_resources()
        sys.exit(1)
    
    finally:
        cleanup_resources()