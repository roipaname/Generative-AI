import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Create a tensor
device = xm.xla_device()
t1 = torch.ones(2, 2, device=device)
t2 = torch.ones(2, 2, device=device)
print("TPU Device:", device)

# Add tensors
t3 = t1 + t2
print("Result:", t3.cpu())
