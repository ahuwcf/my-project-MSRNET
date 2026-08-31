import torch
import clip

# 设置设备（CPU 或 GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载 CLIP 模型（ViT-B/32）
model, preprocess = clip.load("ViT-B/32", device=device)

# 保存模型权重
weight_path = "ViT-B-32.pt"
torch.save(model.state_dict(), weight_path)
print(f"ViT-B-32 权重已保存到: {weight_path}")