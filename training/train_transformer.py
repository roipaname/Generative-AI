import os
# Set environment variables before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["TPU_LIBRARY_PATH"] = "/lib/libtpu.so"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
import time

import torch_xla
import torch_xla.core.xla_model as xm
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
batch_size = 8  # Reduced for stability
epochs = 3
lr = 3e-4

# Initialize tokenizer globally
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class QADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        # Limit dataset size for testing
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(dataset))))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
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
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a dummy sample
            return {
                'input_ids': torch.zeros(max_len, dtype=torch.long),
                'attention_mask': torch.zeros(max_len, dtype=torch.long),
                'start_positions': torch.tensor(0, dtype=torch.long),
                'end_positions': torch.tensor(0, dtype=torch.long)
            }

def main():
    try:
        print("=== Starting TPU Training ===")
        print(f"PyTorch version: {torch.__version__}")
        
        # Initialize TPU
        device = xm.xla_device()
        print(f"TPU device: {device}")
        
        # Get world size
        world_size = xm.xrt_world_size()
        print(f"World size: {world_size}")
        
        if world_size == 0:
            print("❌ No TPU cores detected!")
            print("Try: sudo reboot")
            return

        print("✓ TPU initialized successfully")

        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset("squad")
        train_dataset = QADataset(dataset["train"], tokenizer, max_samples=500)  # Small test
        print(f"Dataset size: {len(train_dataset)}")
        
        # Create DataLoader (no distributed sampler for single process)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=False
        )
        
        # Wrap with TPU parallel loader
        para_loader = pl.MpDeviceLoader(train_loader, device)
        print("✓ DataLoader created")

        # Initialize model
        print("Initializing model...")
        model = QA_TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        model = model.to(device)
        print("✓ Model moved to TPU")
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=0.01,
            eps=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
        print("✓ Optimizer and loss function initialized")

        # Initialize wandb
        wandb.init(
            project="transformer-qa-tpu", 
            name="qa-training-single-process",
            config={
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "world_size": world_size
            }
        )
        print("✓ Wandb initialized")

        # Training loop
        print("Starting training...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            for i, batch in enumerate(para_loader):
                try:
                    start_time = time.time()
                    
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    start_positions = batch['start_positions']
                    end_positions = batch['end_positions']

                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    start_logits, end_logits = model(input_ids, mask=attention_mask)
                    
                    # Calculate loss
                    loss_start = criterion(start_logits, start_positions)
                    loss_end = criterion(end_logits, end_positions)
                    loss = (loss_start + loss_end) / 2

                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # TPU-specific optimizer step
                    xm.optimizer_step(optimizer)
                    
                    # Mark step for TPU
                    xm.mark_step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    batch_time = time.time() - start_time

                    # Log progress
                    if i % 10 == 0:
                        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                        print(f"  Step {i:3d}: Loss={loss.item():.4f}, Avg={avg_loss:.4f}, Time={batch_time:.2f}s")
                        
                        wandb.log({
                            "step_loss": loss.item(),
                            "avg_loss": avg_loss,
                            "epoch": epoch+1,
                            "step": i,
                            "batch_time": batch_time
                        })

                    # Break early for testing
                    if i >= 50:  # Only run 50 steps per epoch for testing
                        break

                except Exception as e:
                    print(f"  Error in batch {i}: {str(e)}")
                    continue

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
            
            wandb.log({
                "epoch_loss": avg_epoch_loss,
                "epoch": epoch+1
            })

        # Save checkpoint
        print("Saving checkpoint...")
        try:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = "checkpoints/qa_transformer_tpu_single.pt"
            
            # Save model state dict
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Checkpoint saved to {checkpoint_path}")
            
            # Save to wandb
            wandb.save(checkpoint_path)
            
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

        print("🎉 Training completed successfully!")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'wandb' in globals():
            wandb.finish()

if __name__ == "__main__":
    main()