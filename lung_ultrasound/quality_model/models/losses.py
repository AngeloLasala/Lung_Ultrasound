import torch
import torch.nn as nn


## CON ONE-HOT
device = "cuda" if torch.cuda.is_available() else "cpu"

# logits (output modello)
B, C, H, W = 2, 2, 256, 256
logits = torch.randn(B, C, H, W, device=device)

# target one-hot
masks = torch.zeros_like(logits)
masks[:,0] = torch.randint(0,2,(B,H,W), device=device)  # coste
masks[:,1] = torch.randint(0,2,(B,H,W), device=device)  # pleura

# pos_weight per canale: coste=1, pleura rara=10
pos_weight = torch.tensor([[1.0], [10.0]], device=device).unsqueeze(-1)

print(logits.shape, masks.shape, pos_weight.shape)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

loss = criterion(logits, masks)
print(loss)


# SENZA ONE-HOT
device = "cuda" if torch.cuda.is_available() else "cpu"

B, C, H, W = 2, 3, 256, 256

# logits grezzi del modello
logits = torch.randn(B, C, H, W, device=device)

# target: 0=background, 1=pleura, 2=coste
target = torch.randint(0, C, (B, H, W), device=device).long()

# pesi per classe: background=1, pleura rara=10, coste=2
class_weights = torch.tensor([1.0, 10.0, 2.0], device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

loss = criterion(logits, target)

print("Loss:", loss.item())