import os
import signal
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
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
device_count = xm.xrt_world_size()
print(f"TPU cores available: {device_count}")



print("Testing TPU connectivity...")
try:
    device_count = xm.xrt_world_size()
    print(f"TPU devices available: {device_count}")
    device = xm.xla_device()
    print(f"Current device: {device}")
except Exception as e:
    print(f"TPU connection error: {e}")
    sys.exit(1)


print("Testing model imports...")
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.transformer.QA_transformer import QA_TransformerModel
    print("Model import successful")
except Exception as e:
    print(f"Model import error: {e}")
    sys.exit(1)


# Enhanced Hyperparameters for longer, better training
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 16  # Optimized for TPU memory
epochs = 10  # Increased for better training
lr = 2e-4  # Slightly reduced for stability
warmup_steps = 1000  # Warmup for better convergence
weight_decay = 0.01  # L2 regularization
max_grad_norm = 1.0  # Gradient clipping
save_every_n_epochs = 2  # Save checkpoints more frequently
eval_every_n_steps = 100  # Evaluate more frequently


class EnhancedQADataset(torch.utils.data.Dataset):
    """Enhanced dataset with proper answer span extraction"""
    
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
        
        # Get the first answer for training
        if answers['text'] and len(answers['text']) > 0:
            answer_text = answers['text'][0]
            answer_start = answers['answer_start'][0] if answers['answer_start'] else 0
        else:
            answer_text = ""
            answer_start = 0

        # Tokenize question and context
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation='only_second',
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Find answer span in tokenized text
        start_positions = 0
        end_positions = 0
        
        if answer_text and answer_start >= 0:
            # Find the token positions that correspond to the answer
            offset_mapping = encoding['offset_mapping'].squeeze(0)
            
            for i, (start_offset, end_offset) in enumerate(offset_mapping):
                if start_offset <= answer_start < end_offset:
                    start_positions = i
                    break
            
            answer_end = answer_start + len(answer_text)
            for i, (start_offset, end_offset) in enumerate(offset_mapping):
                if start_offset < answer_end <= end_offset:
                    end_positions = i
                    break
            
            # Ensure end >= start
            if end_positions < start_positions:
                end_positions = start_positions

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(start_positions, dtype=torch.long),
            'end_positions': torch.tensor(end_positions, dtype=torch.long)
        }


def simple_train_fn(rank):
    """Simplified training function for debugging"""
    print(f"Process {rank} starting...")

    try:
        device = xm.xla_device()
        print(f"Process {rank} using device: {device}")

        print(f"Process {rank}: Creating model...")
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        print(f"Process {rank}: Model created successfully")

        print(f"Process {rank}: Testing tensor operations...")
        test_tensor = torch.randint(0, vocab_size, (2, max_len), dtype=torch.long).to(device)
        output = model(test_tensor)

        print(f"Process {rank}: Tensor operations successful")
        print(f"Process {rank}: Completed successfully")

    except Exception as e:
        print(f"Process {rank} error: {e}")
        traceback.print_exc()


def evaluate_model(model, eval_loader, criterion, device, rank):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            start_logits, end_logits = model(input_ids, mask=attention_mask)
            
            loss_start = criterion(start_logits, start_positions)
            loss_end = criterion(end_logits, end_positions)
            loss = (loss_start + loss_end) / 2
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 20:  # Limit eval batches for speed
                break
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def full_train_fn(rank):
    """Enhanced full training function with better practices"""
    print(f"Process {rank} starting enhanced training...")

    try:
        device = xm.xla_device()

        print(f"Process {rank}: Loading tokenizer and dataset...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_prefix_space=True)

        # Load larger dataset for better training
        train_dataset_raw = load_dataset("squad", split="train[:5000]")  # Increased dataset size
        eval_dataset_raw = load_dataset("squad", split="validation[:500]")

        train_dataset = EnhancedQADataset(train_dataset_raw, tokenizer, max_len)
        eval_dataset = EnhancedQADataset(eval_dataset_raw, tokenizer, max_len)

        world_size = xm.xrt_world_size()
        
        # Training data loader
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

        # Evaluation data loader
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=per_core_batch_size,
            sampler=eval_sampler,
            drop_last=False,
            num_workers=0
        )
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        train_device_loader = MpDeviceLoader(train_loader, device)
        eval_device_loader = MpDeviceLoader(eval_loader, device)


        print(f"Process {rank}: Dataset loaded - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Initialize model with better initialization
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        
        # Enhanced optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()

        print(f"Process {rank}: Starting enhanced training for {epochs} epochs...")

        # Training metrics tracking
        training_history = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'epochs': []
        }

        best_eval_loss = float('inf')
        global_step = 0

        for epoch in range(epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            epoch_losses = []
            epoch_start_time = time.time()

            print(f"Process {rank}: Starting epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(train_device_loader):
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
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                xm.optimizer_step(optimizer)
                scheduler.step()

                epoch_losses.append(loss.item())
                global_step += 1

                # Periodic evaluation and logging
                if global_step % eval_every_n_steps == 0 and xm.is_master_ordinal():
                    current_lr = scheduler.get_last_lr()[0]
                    avg_loss = sum(epoch_losses[-10:]) / min(len(epoch_losses), 10)
                    print(f"Process {rank} Step {global_step} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

                # Periodic evaluation
                if global_step % (eval_every_n_steps * 2) == 0:
                    eval_loss = evaluate_model(model, eval_device_loader, criterion, device, rank)
                    if xm.is_master_ordinal():
                        print(f"Process {rank} Step {global_step} - Eval Loss: {eval_loss:.4f}")
                    
                    # Save best model
                    if eval_loss < best_eval_loss and xm.is_master_ordinal():
                        best_eval_loss = eval_loss
                        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        best_model_path = os.path.join(checkpoint_dir, 'qa_transformer_best.pt')
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch,
                            'step': global_step,
                            'eval_loss': eval_loss,
                        }, best_model_path)
                        print(f"New best model saved with eval loss: {eval_loss:.4f}")

            xm.mark_step()

            # End of epoch processing
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            
            if xm.is_master_ordinal():
                print(f"Process {rank} Epoch {epoch+1} completed in {epoch_time:.2f}s - Avg Loss: {avg_epoch_loss:.4f}")

                # Record training history
                training_history['train_losses'].append(avg_epoch_loss)
                training_history['learning_rates'].append(scheduler.get_last_lr()[0])
                training_history['epochs'].append(epoch + 1)

            # Final evaluation for the epoch
            eval_loss = evaluate_model(model, eval_loader, criterion, device, rank)
            if xm.is_master_ordinal():
                training_history['eval_losses'].append(eval_loss)
                print(f"Process {rank} Epoch {epoch+1} - Final Eval Loss: {eval_loss:.4f}")

            # Save checkpoint every N epochs
            if (epoch + 1) % save_every_n_epochs == 0 and xm.is_master_ordinal():
                checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'qa_transformer_epoch_{epoch+1}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'step': global_step,
                    'train_loss': avg_epoch_loss,
                    'eval_loss': eval_loss,
                    'training_history': training_history
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        if xm.is_master_ordinal():
            print("Enhanced training completed successfully!")

            # Save final model and training history
            checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Final model checkpoint
            final_checkpoint_path = os.path.join(checkpoint_dir, 'qa_transformer_final.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epochs,
                'step': global_step,
                'training_history': training_history,
                'hyperparameters': {
                    'vocab_size': vocab_size,
                    'max_len': max_len,
                    'd_model': d_model,
                    'num_heads': num_heads,
                    'd_ff': d_ff,
                    'num_layers': num_layers,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'lr': lr,
                    'weight_decay': weight_decay
                }
            }, final_checkpoint_path)
            print(f"Final model checkpoint saved: {final_checkpoint_path}")

            # Save training history as JSON
            history_path = os.path.join(checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            print(f"Training history saved: {history_path}")

            # Save model state dict for inference (backward compatibility)
            inference_path = os.path.join(checkpoint_dir, 'qa_transformer_checkpoint.pt')
            torch.save(model.state_dict(), inference_path)
            print(f"Inference model saved: {inference_path}")

    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    print("Starting enhanced TPU training...")
    print(f"Training configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Warmup steps: {warmup_steps}")
    print(f"  - Max gradient norm: {max_grad_norm}")

    print("\n=== Testing simple TPU functionality ===")
    try:
        xmp.spawn(simple_train_fn, args=(), nprocs=8, start_method='fork')
        print("Simple test passed!")
    except Exception as e:
        print(f"Simple test failed: {e}")
        sys.exit(1)

    print("\n=== Starting enhanced training ===")
    try:
        start_time = datetime.now()
        print(f"Training started at: {start_time}")
        
        xmp.spawn(full_train_fn, args=(), nprocs=1, start_method='fork')
        
        end_time = datetime.now()
        total_time = end_time - start_time
        print(f"Enhanced training completed at: {end_time}")
        print(f"Total training time: {total_time}")
        print("Enhanced training test passed!")
        
    except Exception as e:
        print(f"Enhanced training failed: {e}")
        traceback.print_exc()
