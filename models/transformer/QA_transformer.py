import torch.nn as nn
from models.transformer.transformer import TransformerModel

class QA_TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=512):
        super().__init__()
        self.transformer = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.start_classifier = nn.Linear(d_model, 1)
        self.end_classifier = nn.Linear(d_model, 1)
    
    def forward(self, input_ids, mask=None):
        # transformer output shape: (batch, seq_len, d_model)
        hidden_states = self.transformer(input_ids, mask=mask, return_hidden=True)

        
        start_logits = self.start_classifier(hidden_states).squeeze(-1)  # (B, L)
        end_logits = self.end_classifier(hidden_states).squeeze(-1)      # (B, L)
        
        return start_logits, end_logits

