# -*- coding: utf-8 -*-
"""MSRNet 基准测试：Params / FLOPs / FPS
配置与 train.py 一致：Houston13 (48 bands, 7 classes), patch_size=14, embed_dim=512, num_blocks=4
"""
import sys, os, json, time
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录
sys.path.insert(0, BASE)

import clip
from clip.model import build_model

# 优先使用仓库内的 ViT-B-32.pt (state_dict 版本); 否则用官方 clip.load 自动下载
_ckpt = os.path.join(BASE, 'ViT-B-32.pt')
if os.path.exists(_ckpt):
    _sd = torch.load(_ckpt, map_location='cpu')
    clip.load = lambda name='ViT-B/32', device='cuda', jit=False, download_root=None: (
        build_model(_sd).to(device).eval().float(), None)

import MSRNet

BANDS, N_CLASSES, EMBED, BLOCKS = 48, 7, 512, 4

model = MSRNet.MSRNet(embed_dim=EMBED, bands=BANDS, num_classes=N_CLASSES,
                      num_blocks=BLOCKS, clip_model_name='ViT-B/32').cuda()
model.eval()

# ================= 1. Params =================
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
clip_total = sum(p.numel() for p in model.text_encoder.clip_model.parameters())
clip_visual = sum(v.numel() for k, v in model.text_encoder.clip_model.state_dict().items() if k.startswith('visual.'))
clip_text = clip_total - clip_visual

print("=" * 60)
print("Params 统计")
print("=" * 60)
print(f"总参数量 (含完整冻结 CLIP):        {total/1e6:.2f} M")
print(f"可训练参数量:                      {trainable/1e6:.2f} M")
print(f"冻结 CLIP 整体 (含未用的 visual):  {clip_total/1e6:.2f} M")
print(f"冻结 CLIP 文本部分 (实际使用):     {clip_text/1e6:.2f} M")
print(f"自身模块 (MSRNet, 不含 CLIP):      {(total-clip_total)/1e6:.2f} M")
print(f"常用报告口径 = 自身 + CLIP文本:     {(total-clip_visual)/1e6:.2f} M")

# ================= 输入构造 =================
desc = json.load(open(os.path.join(BASE, 'class_descriptions_houston.json')))
phrases = desc[0]              # 该类别全部细粒度短语
global_desc = ' '.join(desc[0][:3])  # 全局类别描述

def make_inputs(bs):
    image = torch.randn(bs, BANDS, 14, 14).cuda()
    text_1 = [list(phrases) for _ in range(bs)]
    text_2 = [global_desc for _ in range(bs)]
    label = torch.zeros(bs, dtype=torch.long).cuda()
    return image, text_1, text_2, label

# 预 tokenize text_2（forward 支持传 tensor）
tok2 = clip.tokenize([global_desc], truncate=True).cuda()

# ================= 2. FLOPs (torch profiler) =================
from torch.profiler import profile, ProfilerActivity

def count_flops(fn):
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
            fn()
    return sum(e.flops for e in prof.key_averages())

# 完整前向（batch=1，含文本编码 —— 文本 prompt 依赖图像特征，每次推理都要算）
img1, t1, _, lab1 = make_inputs(1)
flops_full = count_flops(lambda: model(img1, t1, tok2, lab1))
# 纯视觉分支
img_branch = lambda: model.visual(img1.float())
flops_visual = count_flops(img_branch)

# thop 交叉验证（Conv/Linear/BN 的 MACs）
try:
    from thop import profile as thop_profile
    import copy
    vis_cpu = copy.deepcopy(model.visual).cpu().eval()
    macs_v, _ = thop_profile(vis_cpu, inputs=(img1.cpu(),), verbose=False)
    del vis_cpu
    print(f"[thop 交叉验证] 视觉分支 MACs = {macs_v/1e9:.3f} G (thop) vs {flops_visual/2/1e9:.3f} G (profiler/2)")
except Exception as e:
    print(f"[thop 交叉验证跳过: {e}]")
print(f"[info] 每个样本的细粒度短语数量: {len(phrases)}")

print("=" * 60)
print("FLOPs 统计 (batch=1)")
print("=" * 60)
print(f"完整前向 (含CLIP文本编码):  {flops_full/1e9:.3f} GFLOPs (= {flops_full/2/1e9:.3f} GMACs)")
print(f"视觉分支:                   {flops_visual/1e9:.3f} GFLOPs (= {flops_visual/2/1e9:.3f} GMACs)")
print(f"文本/融合等其他部分:        {(flops_full-flops_visual)/1e9:.3f} GFLOPs")

# ================= 3. FPS =================
def measure_fps(bs, n_iter=100, warmup=20):
    image, text_1, text_2, label = make_inputs(bs)
    tok = clip.tokenize(text_2, truncate=True).cuda()
    with torch.no_grad():
        for _ in range(warmup):
            model(image, text_1, tok, label)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            model(image, text_1, tok, label)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter
    return dt, bs / dt

print("=" * 60)
print("FPS 统计 (RTX 4090, FP32, end-to-end)")
print("=" * 60)
for bs in [1, 64]:
    dt, fps = measure_fps(bs)
    print(f"batch_size={bs:>3}:  每样本 {dt/bs*1000:.2f} ms,  吞吐 {fps:.1f} samples/s (FPS={fps:.1f})")
