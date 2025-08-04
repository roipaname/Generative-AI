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

# GPU-specific environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA errors

# Import model - adjust path as needed for Colab
# If your model is in a different location, modify this path
try:
    from models.transformer.QA_transformer import QA_TransformerModel
except ImportError:
    print("Warning: Could not import QA_TransformerModel. Please ensure the model file is available.")
    # You may need to upload your model file to Colab or define it here

# Hyperparameters optimized for GPU training
vocab_size = 30522
max_len = 384
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
batch_size = 32  # Increased batch size for GPU (adjust based on GPU memory)
epochs = 10
lr = 2e-4
warmup_steps = 500
weight_decay = 0.01
max_grad_norm = 1.0
save_every_n_epochs = 3
eval_every_n_steps = 50
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
    """Clean up GPU resources"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def check_gpu():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available! Please enable GPU in Colab runtime.")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {device_name}")
    print(f"Total GPU Memory: {memory_total:.1f} GB")
    
    return True

def train_model():
    """GPU training function"""
    try:
        # Check GPU availability
        if not check_gpu():
            return
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Quick connectivity test
        print("Running GPU connectivity test...")
        try:
            model_test = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
            test_tensor = torch.randint(0, vocab_size, (2, max_len), dtype=torch.long).to(device)
            test_output = model_test(test_tensor)
            print(f"GPU connectivity test passed! Output shapes: {[x.shape for x in test_output]}")
            
            # Clean up test objects
            del model_test, test_tensor, test_output
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"GPU connectivity test failed: {e}")
            return

        # Load tokenizer and dataset
        print("Loading datasets...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_prefix_space=True)
        train_data = load_dataset("squad", split=f"train[:{max_train_samples}]")
        eval_data = load_dataset("squad", split=f"validation[:{max_eval_samples}]")

        train_dataset = EnhancedQADataset(train_data, tokenizer, max_len)
        eval_dataset = EnhancedQADataset(eval_data, tokenizer, max_len)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,  # Use some parallel workers for data loading
            pin_memory=True  # Faster GPU transfer
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            pin_memory=True
        )

        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)

        # Initialize model, optimizer, and scheduler
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.CrossEntropyLoss()

        print(f"Starting training loop. {len(train_loader)} batches per epoch...")
        print(f"Total training steps: {total_steps}")
        
        # Check initial GPU memory
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")

        global_step = 0
        best_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            step_count = 0

            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Move batch to GPU
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    start_positions = batch['start_positions'].to(device, non_blocking=True)
                    end_positions = batch['end_positions'].to(device, non_blocking=True)

                    optimizer.zero_grad()
                    
                    # Forward pass
                    start_logits, end_logits = model(input_ids, mask=attention_mask)

                    # Calculate loss
                    start_loss = criterion(start_logits, start_positions)
                    end_loss = criterion(end_logits, end_positions)
                    loss = (start_loss + end_loss) / 2

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    step_count += 1
                    global_step += 1

                    # Logging
                    if global_step % eval_every_n_steps == 0:
                        avg_loss = epoch_loss / step_count if step_count > 0 else 0
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        print(f"Step {global_step}/{total_steps} | Batch {batch_idx}/{len(train_loader)} | "
                              f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | GPU: {gpu_memory:.2f}GB")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"GPU OOM error at batch {batch_idx}. Trying to recover...")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e
                        
                except Exception as batch_e:
                    print(f"Error in batch {batch_idx}: {batch_e}")
                    traceback.print_exc()
                    continue

            # Epoch completed
            avg_epoch_loss = epoch_loss / step_count if step_count > 0 else float('inf')
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s | "
                  f"Avg Loss: {avg_epoch_loss:.4f} | Steps: {step_count}")

            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), "checkpoints/best_model.pt")
                print(f"New best model saved! Loss: {best_loss:.4f}")

            # Save periodic checkpoints
            if (epoch + 1) % save_every_n_epochs == 0:
                ckpt_path = f"checkpoints/model_epoch{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_epoch_loss,
                    'global_step': global_step
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

            # Optional: Run evaluation
            if epoch % 2 == 0:  # Evaluate every 2 epochs
                model.eval()
                eval_loss = 0.0
                eval_steps = 0
                
                print("Running evaluation...")
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        try:
                            input_ids = eval_batch['input_ids'].to(device, non_blocking=True)
                            attention_mask = eval_batch['attention_mask'].to(device, non_blocking=True)
                            start_positions = eval_batch['start_positions'].to(device, non_blocking=True)
                            end_positions = eval_batch['end_positions'].to(device, non_blocking=True)

                            start_logits, end_logits = model(input_ids, mask=attention_mask)
                            start_loss = criterion(start_logits, start_positions)
                            end_loss = criterion(end_logits, end_positions)
                            loss = (start_loss + end_loss) / 2

                            eval_loss += loss.item()
                            eval_steps += 1
                            
                        except Exception as e:
                            print(f"Error in evaluation batch: {e}")
                            continue

                avg_eval_loss = eval_loss / eval_steps if eval_steps > 0 else float('inf')
                print(f"Evaluation Loss: {avg_eval_loss:.4f}")
                
                model.train()  # Switch back to training mode

        # Final save
        final_ckpt_path = "checkpoints/final_transformer_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_loss': best_loss,
            'total_epochs': epochs,
            'hyperparameters': {
                'vocab_size': vocab_size,
                'd_model': d_model,
                'num_heads': num_heads,
                'd_ff': d_ff,
                'num_layers': num_layers,
                'max_len': max_len,
                'batch_size': batch_size,
                'lr': lr
            }
        }, final_ckpt_path)
        
        print(f"\nTraining completed successfully!")
        print(f"Final model saved: {final_ckpt_path}")
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
    
    print("=== GPU Training Configuration ===")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Max train samples: {max_train_samples}")
    print(f"Max eval samples: {max_eval_samples}")
    print(f"Model parameters: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")
    
    print(f"\n=== Starting GPU Training ===")
    start_time = datetime.now()
    
    try:
        train_model()
        
        end_time = datetime.now()
        print(f"\nTraining completed successfully in {end_time - start_time}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        cleanup_resources()
        sys.exit(1)
    
    finally:
        cleanup_resources()
        print("Resources cleaned up.")