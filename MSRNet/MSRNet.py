 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import clip
from timm.models.layers import to_2tuple, trunc_normal_


# ========== 梯度反转层（用于域对抗） ==========
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


# ========== Stage-Aware Modulator (fmod) ==========
class StageAwareModulator(nn.Module):
    """轻量调制网络：由上一推理阶段的条件向量 c 预测调制强度
        λ = σ(fmod(c)) ∈ (0, 1)
    仅作用于全局描述路径的 ITPG prompts（λ·P）。短语路径不经过本模块。
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, c):
        # c: (B, D) -> λ: (B,)
        return torch.sigmoid(self.net(c)).squeeze(-1)


# ========== 可学习 DCT、光谱基、FSSMBlock_V2 等 ==========
class LearnableDCT(nn.Module):
    def __init__(self, in_channels, kernel_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        pad = self.kernel_size // 2
        out = F.conv2d(x, self.weight, stride=1, padding=pad, groups=self.in_channels)
        if out.shape[-1] != W:
            out = out[:, :, :H, :W]
        return out


class LearnableOrthogonalSpectralBasis(nn.Module):
    """可学习光谱正交基 B：将原始光谱特征投影到去相关子空间。
    正交约束（QR 正交化）保证变换后的谱分量线性无关，消除光谱通道间的冗余表示。
    DCT 初始化提供有原则的起点（DCT 基近似一阶马尔可夫信号的 KLT 变换）。
    """
    def __init__(self, in_channels, init='dct', use_orthogonal=False, orthogonalize=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_orthogonal = use_orthogonal
        self.orthogonalize = orthogonalize
        self.weight = nn.Parameter(torch.empty(in_channels, in_channels))
        with torch.no_grad():
            if init == 'dct':
                for i in range(in_channels):
                    for j in range(in_channels):
                        self.weight[i, j] = np.cos(np.pi * i * (j + 0.5) / in_channels)
                self.weight.data = self.weight.data / np.sqrt(in_channels / 2)
                if in_channels == 1:
                    self.weight[0, 0] = 1.0
            else:
                nn.init.orthogonal_(self.weight)
        self.bn = nn.BatchNorm1d(in_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.weight
        if self.orthogonalize:
            # 正交约束：QR 分解将基投影为正交矩阵（可微），
            # 确保谱分量线性无关、无冗余（论文 Sec. 3.1.1）
            w = torch.linalg.qr(w).Q
        x_flat = x.view(B, C, H * W)
        y = torch.matmul(w.t(), x_flat)
        y = y.view(B, C, H, W)
        y = self.bn(y.view(B, C, -1)).view(B, C, H, W)
        y = self.gelu(y)
        return y


class FSSMBlock_V2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels

        c_spa = in_channels // 2
        c_freq = in_channels // 4
        c_spec = in_channels - c_spa - c_freq

        self.conv_spa = nn.Sequential(
            nn.Conv2d(in_channels, c_spa, 1),
            nn.BatchNorm2d(c_spa),
            nn.GELU()
        )
        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels, c_freq, 1),
            nn.BatchNorm2d(c_freq),
            nn.GELU()
        )
        self.conv_spec = nn.Sequential(
            nn.Conv2d(in_channels, c_spec, 1),
            nn.BatchNorm2d(c_spec),
            nn.GELU()
        )

        self.spa_conv3 = nn.Sequential(
            nn.Conv2d(c_spa, c_spa, 3, padding=1, groups=c_spa),
            nn.BatchNorm2d(c_spa),
            nn.GELU()
        )
        self.spa_conv5 = nn.Sequential(
            nn.Conv2d(c_spa, c_spa, 5, padding=2, groups=c_spa),
            nn.BatchNorm2d(c_spa),
            nn.GELU()
        )
        self.spa_conv7 = nn.Sequential(
            nn.Conv2d(c_spa, c_spa, 7, padding=3, groups=c_spa),
            nn.BatchNorm2d(c_spa),
            nn.GELU()
        )
        self.spa_weight = nn.Parameter(torch.ones(3) / 3)
        self.spa_combine = nn.Sequential(
            nn.Conv2d(c_spa, c_spa, 1),
            nn.BatchNorm2d(c_spa),
            nn.GELU()
        )

        self.dct = LearnableDCT(c_freq, kernel_size=8)
        self.dct_bn = nn.BatchNorm2d(c_freq)

        self.spectral_basis = LearnableOrthogonalSpectralBasis(c_spec, init='dct', use_orthogonal=False)
        self.spec_bn = nn.BatchNorm2d(c_spec)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x_spa = self.conv_spa(x)
        x_freq = self.conv_freq(x)
        x_spec = self.conv_spec(x)

        out3 = self.spa_conv3(x_spa)
        out5 = self.spa_conv5(x_spa)
        out7 = self.spa_conv7(x_spa)
        spa_weights = F.softmax(self.spa_weight, dim=0)
        spa_out = spa_weights[0] * out3 + spa_weights[1] * out5 + spa_weights[2] * out7
        spa_out = self.spa_combine(spa_out)

        freq_out = self.dct(x_freq)
        freq_out = self.dct_bn(freq_out)

        spec_out = self.spectral_basis(x_spec)
        spec_out = self.spec_bn(spec_out)

        combined = torch.cat([spa_out, freq_out, spec_out], dim=1)
        attn = self.global_pool(combined)
        attn = F.softmax(attn, dim=1)
        combined = combined * attn
        out = self.final_conv(combined)
        return out


class FSSMImageEncoder(nn.Module):
    """UM3F 多域视觉编码器：空间 / 频率 / 光谱三分支 + 通道注意力重加权。
    输出三个表示：全局图像特征 g（GAP + 线性投影）、
    中间特征图 M（用于局部 patch 提取）、中层池化特征 m（作为文本编码器的图像条件）。
    """
    def __init__(self, in_channels, embed_dim, num_blocks=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(FSSMBlock_V2(embed_dim, embed_dim))
            if (i + 1) % 2 == 0:
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU()
                ))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, text_prompt=None):
        x = x.float()
        x = self.stem(x)
        mid_feat_map = None
        for i, layer in enumerate(self.blocks):
            if isinstance(layer, FSSMBlock_V2) and i == len(self.blocks) - 2:
                mid_feat_map = x
            x = layer(x)
        if mid_feat_map is None:
            mid_feat_map = x
        global_feat = self.global_pool(x).squeeze(-1).squeeze(-1)
        global_feat = self.proj(global_feat)
        mid_feat_pooled = self.global_pool(mid_feat_map).squeeze(-1).squeeze(-1)
        return global_feat, mid_feat_map, mid_feat_pooled


# ========== CLIP 文本编码器相关 ==========
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# ========== ITPG: 图像条件提示生成器 ==========
class ITPG(nn.Module):
    """Image-conditioned Prompt Generator:
        P = reshape(W_itpg · m),  W_itpg ∈ R^{D×(N_itpg·d_clip)}
    输入为中层视觉特征 m，输出 N_itpg 个图像条件 prompt tokens。
    """
    def __init__(self, img_dim, text_dim, num_prompts=4):
        super().__init__()
        self.num_prompts = num_prompts
        self.proj = nn.Linear(img_dim, text_dim * num_prompts)

    def forward(self, img_feat):
        B = img_feat.shape[0]
        prompts = self.proj(img_feat).view(B, self.num_prompts, -1)
        return prompts


# ========== MCL: 模态上下文学习器 ==========
class MCL(nn.Module):
    """Modality Context Learner：从可学习码本中取模态上下文 tokens，
    区分全局描述（modality_id=0）与短语级输入（modality_id=1）。
    """
    def __init__(self, num_modalities, text_dim, context_len=6):
        super().__init__()
        self.context = nn.Parameter(torch.randn(num_modalities, context_len, text_dim))

    def forward(self, modality_id):
        return self.context[modality_id].unsqueeze(0)


class CLIPTextEncoder(nn.Module):
    """CATE: Conditional Adaptive Text Encoder（冻结 CLIP 骨干）。
    全局描述路径: [C; λ·P; T]，其中 C 为 MCL tokens，P 为 ITPG prompts，
    λ = σ(fmod(c)) 为 stage-aware 调制强度（由外部传入）。
    短语路径: 仅使用 MCL（modality_id=1），不使用 ITPG 与调制器。
    """
    def __init__(self, embed_dim=512, clip_model_name='ViT-B/32', num_prompts=4, context_len=6):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name, device='cuda')
        self.clip_model = self.clip_model.float()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.text_dim = self.clip_model.text_projection.shape[1]
        self.embed_dim = embed_dim
        self.itpg = ITPG(img_dim=embed_dim, text_dim=self.text_dim, num_prompts=num_prompts)
        self.mcl = MCL(num_modalities=2, text_dim=self.text_dim, context_len=context_len)
        self.proj = nn.Linear(self.text_dim, embed_dim)

    def forward(self, text_tokens, img_feat, modality_id=0, lam=None):
        img_feat = img_feat.float()
        if not isinstance(text_tokens, torch.Tensor):
            if isinstance(text_tokens, list):
                text_tokens = torch.stack(text_tokens, dim=0)
            else:
                raise TypeError(f"text_tokens must be a tensor or list of tensors, got {type(text_tokens)}")
        with torch.no_grad():
            word_embeds = self.clip_model.token_embedding(text_tokens).float()
            pos_embed = self.clip_model.positional_embedding.float()
            L = word_embeds.shape[1]
            pos_embed = pos_embed[:L, :].unsqueeze(0)

        itpg_tokens = self.itpg(img_feat)
        if lam is not None:
            # Stage-aware 调制: λ · P（论文 Eq. 5 之后的调制项）
            itpg_tokens = itpg_tokens * lam.reshape(-1, 1, 1)
        mcl_tokens = self.mcl(modality_id)
        mcl_tokens = mcl_tokens.expand(img_feat.shape[0], -1, -1)

        max_len = 77
        total_prompt_len = mcl_tokens.shape[1] + itpg_tokens.shape[1]
        if total_prompt_len + L > max_len:
            word_embeds = word_embeds[:, :max_len - total_prompt_len, :]

        # 注意: CLIP 参数已冻结 (requires_grad=False)，但 transformer 前向不能包在
        # no_grad 里——否则梯度无法回传到输入端的可学习 tokens (MCL / λ·P / ITPG)
        x = torch.cat([mcl_tokens, itpg_tokens, word_embeds], dim=1)
        new_len = x.shape[1]
        new_pos_embed = self.clip_model.positional_embedding[:new_len, :].unsqueeze(0)
        x = x + new_pos_embed

        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        text_feat = x[:, -1, :]
        return self.proj(text_feat)


# ========== 多正样本局部对齐损失 ==========
class MultiPositiveLocalAlignment(nn.Module):
    """CDL 多正样本局部对齐（论文 Eq. 11）：
    每个短语选取 top-k_max 最相似的视觉 patch 作为正样本。
    """
    def __init__(self, temperature=0.07, top_k=3):
        super().__init__()
        self.temp = temperature
        self.top_k = top_k

    def forward(self, local_img_feat, phrases_feat_list, phrase_counts):
        B, N, D = local_img_feat.shape
        total_loss = 0.0
        for b in range(B):
            img_b = local_img_feat[b]          # (N, D)
            text_b = phrases_feat_list[b]      # (M_b, D)
            if text_b.shape[0] == 0:
                continue
            sim = torch.matmul(F.normalize(text_b, dim=-1), F.normalize(img_b, dim=-1).t()) / self.temp
            k = min(self.top_k, N)
            topk_sim, _ = torch.topk(sim, k, dim=1)   # (M_b, k)
            pos_exp = torch.exp(topk_sim).sum(dim=1)  # (M_b,)
            all_exp = torch.exp(sim).sum(dim=1)       # (M_b,)
            loss_m = -torch.log(pos_exp / (all_exp + 1e-8))
            total_loss += loss_m.mean()
        return total_loss / B


# ========== 跨模态融合模块（论文 Eq. 6-8） ==========
class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cond_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, visual_feat, text_feat, condition=None):
        B, N, D = visual_feat.shape
        if condition is not None:
            cond = self.cond_proj(condition).unsqueeze(1)  # (B,1,D)
            visual_feat = visual_feat + cond
            text_feat = text_feat + cond
        fused, _ = self.attn(visual_feat, text_feat, text_feat)
        fused = fused + visual_feat
        fused = self.norm(fused)
        fused = fused + self.linear(fused)
        return fused


# ========== 主模型（MSR 三阶段推理） ==========
class MSRNet(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 bands: int = None,
                 num_classes: int = None,
                 num_blocks: int = 4,
                 clip_model_name: str = 'ViT-B/32',
                 top_k: int = 3,
                 domain_weight: float = 0.1,   # δ (Eq. 20)
                 sem_weight: float = 0.1,      # ε (Eq. 20)
                 **kwargs):
        super().__init__()
        # 兼容旧参数名
        if bands is None and 'n_bands' in kwargs:
            bands = kwargs['n_bands']
        if num_classes is None and 'n_classes' in kwargs:
            num_classes = int(kwargs['n_classes'])
        if bands is None or num_classes is None:
            raise ValueError("bands and num_classes must be provided either directly or via kwargs (n_bands, n_classes)")

        self.embed_dim = embed_dim
        self.top_k = top_k
        self.num_classes = num_classes
        self.domain_weight = domain_weight
        self.sem_weight = sem_weight

        # 视觉编码器 (UM3F)
        self.visual = FSSMImageEncoder(in_channels=bands, embed_dim=embed_dim, num_blocks=num_blocks)
        # 文本编码器 (CATE)
        self.text_encoder = CLIPTextEncoder(embed_dim=embed_dim, clip_model_name=clip_model_name)
        # Stage-aware modulator (fmod)
        self.modulator = StageAwareModulator(embed_dim)

        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))

        # MSR 三个阶段的分类器 (Eq. 9 / 12 / 17)
        self.classifier_stage1 = nn.Linear(embed_dim, num_classes)
        self.classifier_stage2 = nn.Linear(embed_dim, num_classes)
        self.classifier_stage3 = nn.Linear(embed_dim, num_classes)

        # 跨模态融合模块
        self.cross_fusion = CrossModalFusion(embed_dim, num_heads=8)

        # 域判别器 (Eq. 14): 两层，GRL 之后
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2)
        )

        # 损失组件
        self.multi_positive_loss = MultiPositiveLocalAlignment(temperature=0.07, top_k=top_k)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.5)).float()

    def encode_phrases_from_text1(self, text_1):
        """短语级文本编码：仅使用 MCL（modality_id=1）指示短语级输入，
        不使用 ITPG prompts 与 stage-aware 调制器，保持局部语义纯度。
        """
        device = next(self.parameters()).device
        phrases_feat_list = []
        phrase_counts = []
        n_mcl = self.text_encoder.mcl.context.shape[1]
        for sample_texts in text_1:
            unique_phrases = list(OrderedDict.fromkeys(sample_texts))
            if len(unique_phrases) == 0:
                phrases_feat_list.append(torch.zeros(0, self.embed_dim).to(device))
                phrase_counts.append(0)
                continue
            tokens = []
            for phrase in unique_phrases:
                token = clip.tokenize(phrase, truncate=True).to(device)
                tokens.append(token)
            tokens = torch.cat(tokens, dim=0)  # (n, L)
            word_embeds = self.text_encoder.clip_model.token_embedding(tokens).float()  # (n, L, d)
            L = word_embeds.shape[1]
            if n_mcl + L > 77:
                word_embeds = word_embeds[:, :77 - n_mcl, :]
                L = word_embeds.shape[1]
            # MCL 上下文 tokens（短语级模态, id=1）；无 ITPG、无 λ 调制
            mcl_tokens = self.text_encoder.mcl(1).expand(word_embeds.shape[0], -1, -1)
            x = torch.cat([mcl_tokens, word_embeds], dim=1)  # [C; T]
            x = x + self.text_encoder.clip_model.positional_embedding[:n_mcl + L, :].unsqueeze(0)
            x = x.permute(1, 0, 2)
            # 不加 no_grad: 保证 MCL tokens 可学习（CLIP 参数本身已冻结）
            x = self.text_encoder.clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            eos_positions = tokens[:, :L].argmax(dim=-1)
            text_feat = x[torch.arange(x.shape[0]), n_mcl + eos_positions]
            phrase_feat = self.text_encoder.proj(text_feat)
            phrases_feat_list.append(phrase_feat)
            phrase_counts.append(len(unique_phrases))
        return phrases_feat_list, phrase_counts

    def compute_mutual_loss(self, global_feat, local_agg_feat, labels, temperature=0.1):
        """全局-局部跨尺度一致性损失（互信息形式）"""
        global_norm = F.normalize(global_feat, dim=-1)
        local_norm = F.normalize(local_agg_feat, dim=-1)
        logit_scale = torch.exp(torch.ones([]) * np.log(1 / temperature)).to(global_feat.device)
        sim = logit_scale * global_norm @ local_norm.t()
        loss = F.cross_entropy(sim, labels)
        loss_t = F.cross_entropy(sim.t(), labels)
        return (loss + loss_t) / 2

    def forward(self, image, text_1, text_2, label, text_ratio=0.1, domain_label=None, stage='msr'):
        """
        MSR 三阶段顺序推理：
          Stage 1 (SCL): 粗粒度语义初始化，λ1 = σ(fmod(g))
          Stage 2 (CDL): 细粒度局部精炼，条件 c(1) = g（短语路径无调制）
          Stage 3 (DIL): 跨域语义校正，λ3 = σ(fmod(r))，域感知文本 t3 + L_sem
        Args:
            image: (B, C, H, W)
            text_1: list of list of str, 细粒度短语
            text_2: str or tensor, 全局类别描述
            label: (B,) 类别标签
            domain_label: (B,) 域标签（0源域，1目标域），可选
        Returns:
            training=True: (total_loss, final_logits, loss_dict)
            inference: (0, final_logits, loss_dict)
        """
        if stage != 'msr':
            raise ValueError("This model only supports stage='msr'. Joint mode has been removed.")

        B = image.shape[0]
        image = image.float()
        label = label.long()

        # 1. 公共特征提取（g, M, m）
        img_global, img_local_map, img_mid_pooled = self.visual(image)
        img_local_feat = self.local_pool(img_local_map)
        _, C, Hp, Wp = img_local_feat.shape
        num_patches = Hp * Wp
        img_local = img_local_feat.reshape(B, C, num_patches).permute(0, 2, 1)  # (B, N, D)

        # 2. Stage 1 文本特征提取（全局描述，λ1 = σ(fmod(g)), 条件 c = g）
        global_tokens = None
        lam1 = None
        if text_2 is None:
            global_text_feat = torch.zeros_like(img_global)
        else:
            if not isinstance(text_2, torch.Tensor):
                global_tokens = clip.tokenize(text_2, truncate=True).to(image.device)
            else:
                global_tokens = text_2
            lam1 = self.modulator(img_global)  # SCL: 高 λ1，文本高度适配视觉场景
            global_text_feat = self.text_encoder(global_tokens, img_mid_pooled,
                                                 modality_id=0, lam=lam1)

        # 3. 短语特征（仅 MCL，无 ITPG / 调制）
        phrases_feat_list, phrase_counts = [], []
        if text_1 is not None:
            phrases_feat_list, phrase_counts = self.encode_phrases_from_text1(text_1)

        loss_dict = {}
        total_loss = 0.0
        if lam1 is not None:
            loss_dict['lambda_1'] = lam1.mean().item()

        # 4. Stage 1 (SCL): 粗粒度语义初始化
        logits_stage1 = self.classifier_stage1(img_global)
        loss_stage1 = F.cross_entropy(logits_stage1, label)
        total_loss += loss_stage1
        loss_dict['loss_stage1'] = loss_stage1.item()
        condition = img_global.detach()  # c(1) = g (Eq. 10)

        # 5. 全局对比损失（HACD）
        img_norm = F.normalize(img_global, dim=-1)
        text_norm = F.normalize(global_text_feat, dim=-1)
        logit_scale = self.logit_scale.exp()
        logits_global = logit_scale * img_norm @ text_norm.t()
        loss_global = (F.cross_entropy(logits_global, label) + F.cross_entropy(logits_global.t(), label)) / 2
        total_loss += 0.2 * loss_global
        loss_dict['loss_global'] = loss_global.item()

        # 6. Stage 2 (CDL): 细粒度局部精炼，条件 c(1) = g
        max_phrases = max(phrase_counts) if phrase_counts else 0
        if max_phrases > 0 and text_1 is not None:
            # 短语填充对齐
            text_feat_batch = []
            mask = []
            for i, feat in enumerate(phrases_feat_list):
                if feat.shape[0] == 0:
                    pad = torch.zeros(max_phrases, self.embed_dim).to(feat.device)
                    mask_i = torch.zeros(max_phrases, dtype=torch.bool).to(feat.device)
                else:
                    pad = torch.cat([feat, torch.zeros(max_phrases - feat.shape[0], self.embed_dim).to(feat.device)],
                                    dim=0)
                    mask_i = torch.cat([torch.ones(feat.shape[0], dtype=torch.bool).to(feat.device),
                                        torch.zeros(max_phrases - feat.shape[0], dtype=torch.bool).to(feat.device)],
                                       dim=0)
                text_feat_batch.append(pad)
                mask.append(mask_i)
            text_feat_batch = torch.stack(text_feat_batch, dim=0)  # (B, M_max, D)
            mask = torch.stack(mask, dim=0)

            # 条件引导的跨模态融合
            fused_local = self.cross_fusion(img_local, text_feat_batch, condition=condition)
            agg_local = fused_local.mean(dim=1)  # r = (1/N) Σ V_out (Eq. 12)

            logits_stage2 = self.classifier_stage2(agg_local)
            loss_stage2 = F.cross_entropy(logits_stage2, label)
            total_loss += loss_stage2
            loss_dict['loss_stage2'] = loss_stage2.item()

            # 多正样本局部对齐损失（Top-K, Eq. 11）
            if sum(phrase_counts) > 0:
                loss_multi = self.multi_positive_loss(img_local, phrases_feat_list, phrase_counts)
                total_loss += 0.5 * loss_multi
                loss_dict['loss_multi_positive'] = loss_multi.item()

            condition = agg_local.detach()  # c(2) = r (Eq. 13)
        else:
            logits_stage2 = logits_stage1
            loss_stage2 = loss_stage1
            condition = img_global.detach()

        # 7. 跨尺度一致性（g 与 r 之间）
        loss_mutual = self.compute_mutual_loss(img_global, condition, label, temperature=0.1)
        total_loss += 0.3 * loss_mutual
        loss_dict['loss_mutual'] = loss_mutual.item()

        # 8. Stage 3 (DIL): 跨域语义校正
        if domain_label is not None:
            reversed_feat = grad_reverse(condition, alpha=1.0)
            domain_pred = self.domain_classifier(reversed_feat)
            loss_domain = F.cross_entropy(domain_pred, domain_label)  # Eq. 15
            total_loss += self.domain_weight * loss_domain
            loss_dict['loss_domain'] = loss_domain.item()

            # 域感知文本特征 t3: λ3 = σ(fmod(c(2)=r))，中间强度λ3 (Eq. 16 前置)
            if global_tokens is not None:
                lam3 = self.modulator(condition)
                t3 = self.text_encoder(global_tokens, img_mid_pooled, modality_id=0, lam=lam3)
                loss_sem = (1 - F.cosine_similarity(condition, t3, dim=-1)).mean()  # Eq. 16
                total_loss += self.sem_weight * loss_sem
                loss_dict['loss_sem'] = loss_sem.item()
                loss_dict['lambda_3'] = lam3.mean().item()

            logits_stage3 = self.classifier_stage3(condition)  # Eq. 17
            loss_stage3 = F.cross_entropy(logits_stage3, label)  # Eq. 19
            total_loss += loss_stage3
            loss_dict['loss_stage3'] = loss_stage3.item()
            final_logits = logits_stage3
        else:
            # 域标签不可用时：省略对抗域损失，直接用 CDL 输出预测
            final_logits = logits_stage2

        if self.training:
            return total_loss, final_logits, loss_dict
        else:
            return torch.tensor(0), final_logits, loss_dict

 
