import torch
import torch.nn as nn
import math



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # From [B, L] to [B, 1, 1, L]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
class GPTDatatsetV1(Dataset):
  def __init__(self,txt,tokenizer,max_length,stride) :
    self.input_ids=[]
    self.target_ids=[]
    token_ids=tokenizer.encode(txt,allowed_special={'<|endoftext|>'})
    for i in range(0,len(token_ids)-max_length,stride):
      self.input_chunk=token_ids[i:i+max_length]
      self.target_chunk= token_ids[i+1:i+max_length+1]
      self.input_ids.append(torch.tensor(self.input_chunk))
      self.target_ids.append(torch.tensor(self.target_chunk))
  def __len__(self):
    return len(self.input_ids)
  def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
  tokenizer=tiktoken.get_encoding("gpt2")
  dataset=GPTDatatsetV1(txt,tokenizer,max_length,stride)
  dataloader=DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=True
  )
  return dataloader

with open("small-text-sample.txt",'r',encoding="utf-8") as f:
  raw_text=f.read()
tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)

vocab_size = 50257
output_dim = 256
max_len = 1024
context_length = max_len


token_embedding_layer = nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length)


import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a tensor with 7 rows and 3 columns for x, y, z coordinates
input = torch.tensor([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9]
], dtype=torch.float)

words = ["your", "journey", "starts", "here", "go", "for", "it"]

# Split into x, y, z
x_coords = input[:,0].numpy()
y_coords = input[:, 1].numpy()
z_coords = input[:, 2].numpy()
print(x_coords)
print(y_coords)
print(z_coords)
# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for x, y, z, word in zip(x_coords, y_coords, z_coords, words):
    ax.scatter(x, y, z, color='blue')
    ax.text(x, y, z, word, color='red')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Word Plot")
plt.show()



