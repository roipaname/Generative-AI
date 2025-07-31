import torch
import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.ff(x)