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

# Critical: Environment setup BEFORE any torch_xla imports
os.environ["XLA_USE_SPMD"] = "1"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["XLA_PERSISTENT_CACHE_DEVICE"] = "1"
os.environ["XLA_CACHE_SIZE"] = "128MB"

# Import model after appending project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer.QA_transformer import QA_TransformerModel

# Hyperparameters optimized for single TPU core
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 32  # Reduced for single core
epochs = 10
lr = 2e-4
warmup_steps = 500
weight_decay = 0.01
max_grad_norm = 1.0
save_every_n_epochs = 3
eval_every_n_steps = 50  # More frequent logging
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
    """Clean up TPU resources"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def train_fn(index):
    """Single TPU core training function"""
    try:
        # Initialize XLA device
        device = xm.xla_device()
        print(f"Initialized on device: {device}")
        
        # Quick connectivity test
        print("Running connectivity test...")
        model_test = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        test_tensor = torch.randint(0, vocab_size, (2, max_len), dtype=torch.long).to(device)
        test_output = model_test(test_tensor)
        print(f"Connectivity test passed! Output shapes: {[x.shape for x in test_output]}")
        
        # Clean up test objects
        del model_test, test_tensor, test_output
        
        # Load datasets and tokenizer
        print("Loading datasets...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_prefix_space=True)
        train_data = load_dataset("squad", split=f"train[:{max_train_samples}]")
        eval_data = load_dataset("squad", split=f"validation[:{max_eval_samples}]")

        train_dataset = EnhancedQADataset(train_data, tokenizer, max_len)
        eval_dataset = EnhancedQADataset(eval_data, tokenizer, max_len)
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        os.makedirs("checkpoints", exist_ok=True)

        # Create simple data loader (no distributed sampling needed for single core)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=False
        )

        # Initialize model and training components
        total_params = sum(p.numel() for p in QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).parameters())
        print(f"Initializing model with {total_params:,} parameters...")
        
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.CrossEntropyLoss()

        print("Starting training loop...")
        print(f"Steps per epoch: {len(train_loader)}")
        print(f"Total training steps: {total_steps}")
        print(f"Batch size: {batch_size}")
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            step_count = 0
            
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            epoch_start_time = time.time()
            
            # Simple iteration without ParallelLoader
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)

                    optimizer.zero_grad()
                    start_logits, end_logits = model(input_ids, mask=attention_mask)
                    
                    start_loss = criterion(start_logits, start_positions)
                    end_loss = criterion(end_logits, end_positions)
                    loss = (start_loss + end_loss) / 2
                    
                    loss.backward()
                    
                    # For single TPU core, use regular optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    xm.optimizer_step(optimizer)  # Still use XLA optimizer step
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    step_count += 1
                    global_step += 1

                    # Periodic logging
                    if global_step % eval_every_n_steps == 0:
                        avg_loss = epoch_loss / step_count if step_count > 0 else 0
                        print(f"Step {global_step:4d}/{total_steps} | Batch {batch_idx:3d}/{len(train_loader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
                        sys.stdout.flush()
                
                except Exception as batch_e:
                    print(f"Error in batch {batch_idx}: {batch_e}")
                    traceback.print_exc()
                    continue
            
            # Calculate and report epoch statistics
            epoch_time = time.time() - epoch_start_time
            if step_count > 0:
                avg_epoch_loss = epoch_loss / step_count
                
                print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s | Steps: {step_count} | Avg Loss: {avg_epoch_loss:.4f}")
                
                # Save best model
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    torch.save(model.state_dict(), "checkpoints/best_model.pt")
                    print(f"New best model saved: checkpoints/best_model.pt (Loss: {avg_epoch_loss:.4f})")
                
                # Periodic checkpoint
                if (epoch + 1) % save_every_n_epochs == 0:
                    ckpt_path = f"checkpoints/model_epoch{epoch + 1}.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")
                    
                sys.stdout.flush()
            else:
                print(f"WARNING: Epoch {epoch + 1} had no training steps!")
                
            # Force XLA synchronization between epochs
            xm.mark_step()

        # Save final model
        final_ckpt_path = "checkpoints/final_transformer_model.pt"
        torch.save(model.state_dict(), final_ckpt_path)
        print(f"Final model saved: {final_ckpt_path}")
        print(f"\nTraining completed successfully!")
        print(f"Best loss achieved: {best_loss:.4f}")

    except Exception as e:
        print(f"Training failed: {e}")
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
    
    print("=== Single TPU Core Training Configuration ===")
    print(f"TPU Type: Single Core")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Max samples: {max_train_samples}")
    
    print(f"\n=== Starting Single TPU Training ===")
    start_time = datetime.now()
    
    try:
        # Single TPU core - use nprocs=1
        xmp.spawn(train_fn, nprocs=1, start_method='fork')

        print(f"\nTraining completed successfully in {datetime.now() - start_time}")
        print("Utilized single TPU core")
        
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        cleanup_resources()
        sys.exit(1)
    
    finally:
        cleanup_resources()