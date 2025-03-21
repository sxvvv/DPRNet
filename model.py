import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class CrossAttention(nn.Module):
    def __init__(self, dim, text_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(text_dim, 2*dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, guidance):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(guidance)
        B_, N_, C_ = kv.shape
        kv = kv.reshape(B_, N_, 2, self.num_heads, C_ // self.num_heads // 2).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma

class Prompt(nn.Module):
    def __init__(self, length=64, embed_dim=256, prompt_pool=True, 
                 pool_size=30, top_k=5, batchwise_prompt=True,
                 diversity_weight=0.5, history_size=100):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.diversity_weight = diversity_weight
        self.history_size = history_size

        if prompt_pool:
            # 初始化提示库，使用30个提示
            self.prompt = nn.Parameter(torch.randn(pool_size, length, embed_dim))
            nn.init.uniform_(self.prompt, -1, 1)
            self.prompt_key = nn.Parameter(torch.randn(pool_size, embed_dim))
            nn.init.uniform_(self.prompt_key, -1, 1)
            
            # 初始化历史记录缓冲区 - 100帧历史窗口
            self.register_buffer('history_buffer', torch.zeros(history_size, pool_size))
            self.register_buffer('history_pointer', torch.zeros(1, dtype=torch.long))
            
            # 任务相关性评分网络
            self.relevance_network = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            
            # 自适应选择阈值
            self.register_buffer('base_threshold', torch.tensor(0.5))
            self.register_buffer('complexity_estimate', torch.tensor(0.0))
            self.lambda_scale = 0.1  # 缩放因子

    def update_history(self, selected_indices):
        """更新历史使用记录"""
        # 创建当前选择的one-hot向量
        current_selection = torch.zeros(self.pool_size, device=selected_indices.device)
        current_selection.scatter_(0, selected_indices, 1.0)
        
        # 更新历史缓冲区
        idx = self.history_pointer.item()
        self.history_buffer[idx] = current_selection
        
        # 循环更新指针
        self.history_pointer[0] = (self.history_pointer[0] + 1) % self.history_size

    def compute_diversity_score(self):
        """基于历史使用频率计算多样性得分"""
        # 计算每个提示在历史窗口中的使用频率
        usage_frequency = torch.sum(self.history_buffer, dim=0) / self.history_size
        
        # 多样性得分与使用频率成反比
        diversity_score = 1.0 - usage_frequency
        return diversity_score

    def compute_relevance_score(self, x_embed):
        """计算任务相关性得分"""
        # 提取特征统计量
        x_mean = torch.mean(x_embed, dim=1)  # [B, C]
        
        # 每个提示的任务相关性
        prompt_relevance = []
        for i in range(self.pool_size):
            # 计算每个提示与当前任务的相关性
            prompt_feat = self.prompt_key[i].unsqueeze(0).expand(x_mean.size(0), -1)
            relevance = self.relevance_network(prompt_feat * x_mean)  # 元素级乘法捕获相互作用
            prompt_relevance.append(relevance)
            
        # 将所有提示的相关性拼接并归一化
        relevance_scores = torch.cat(prompt_relevance, dim=1)  # [B, pool_size]
        relevance_scores = F.softmax(relevance_scores, dim=1)
        return relevance_scores

    def compute_dynamic_threshold(self, x_embed):
        """计算动态选择阈值"""
        # 估计退化复杂度 
        feat_std = torch.std(x_embed, dim=(1,2)).mean()
        self.complexity_estimate = 0.9 * self.complexity_estimate + 0.1 * feat_std
        
        # 调整阈值: τ_t = τ_base · exp(-λ·complexity)
        dynamic_threshold = self.base_threshold * torch.exp(-self.lambda_scale * self.complexity_estimate)
        return torch.clamp(dynamic_threshold, min=0.2, max=0.8)

    def forward(self, x_embed, depth_feature):
        batch_size = x_embed.shape[0]
        out = {}

        if self.prompt_pool:
            # 计算特征-提示相似度 (相似性得分)
            x_embed_pooled = torch.mean(x_embed, dim=1)  # [B, C]
            x_embed_norm = F.normalize(x_embed_pooled, p=2, dim=1)
            prompt_norm = F.normalize(self.prompt_key, p=2, dim=1)
            similarity = torch.matmul(x_embed_norm, prompt_norm.T)  # [B, pool_size]
            
            # 计算多样性得分并扩展到批次大小
            diversity = self.compute_diversity_score().unsqueeze(0).expand(batch_size, -1)
            
            # 计算任务相关性得分
            relevance = self.compute_relevance_score(x_embed)
            
            # 多标准融合选择分数
            alpha, beta, gamma = 0.5, 0.3, 0.2  # 可调整的权重
            final_scores = (
                alpha * similarity + 
                beta * diversity + 
                gamma * relevance
            )
            
            # 动态阈值选择
            threshold = self.compute_dynamic_threshold(x_embed)
            mask = final_scores > threshold
            
            # 确保即使所有分数低于阈值也至少选择top_k个提示
            if self.batchwise_prompt:
                # 计算批次级平均分数
                batch_scores = torch.mean(final_scores, dim=0)
                _, selected_idx = torch.topk(batch_scores, k=self.top_k)
                selected_idx = selected_idx.expand(batch_size, -1)
            else:
                # 每个样本独立选择提示
                _, selected_idx = torch.topk(final_scores, k=self.top_k, dim=1)
            
            # 更新历史使用记录
            self.update_history(torch.unique(selected_idx.view(-1)))
            
            # 获取选中的提示并重塑
            if self.batchwise_prompt:
                batched_prompt = self.prompt[selected_idx[0]].unsqueeze(0).expand(batch_size, -1, -1, -1)
            else:
                # 处理每个样本独立选择的情况
                batched_prompt = torch.stack([self.prompt[idx] for idx in selected_idx])
            
            # 重塑为正确的输出格式
            batched_prompt = batched_prompt.reshape(batch_size, -1, self.embed_dim)
            
            # 深度特征与提示交互
            depth_proj = self.depth_transform(depth_feature) if hasattr(self, 'depth_transform') else depth_feature
            depth_attn = torch.matmul(
                F.normalize(depth_proj, p=2, dim=-1),
                F.normalize(batched_prompt.transpose(1, 2), p=2, dim=1)
            ) / math.sqrt(depth_proj.size(-1))
            depth_attn_weights = F.softmax(depth_attn, dim=-1)
            depth_modulated_prompt = torch.matmul(depth_attn_weights, batched_prompt)
            
            # 存储输出
            out['prompted_embedding'] = depth_modulated_prompt
            out['selection_scores'] = final_scores
            out['selected_indices'] = selected_idx
            out['diversity_scores'] = diversity[0]  # 仅存储一个批次的多样性得分
            
        return out

class ImageRestorationModel(nn.Module):
    def __init__(self, img_channel=6, out_channel=3, width=64, middle_blk_num=1,
                 enc_blk_nums=[1,1,2,6], dec_blk_nums=[1,1,1,1],
                 prompt_pool=True, pool_size=10, top_k=5,
                 diversity_weight=0.5, history_size=100):
        super().__init__()
        
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, out_channel, 3, 1, 1)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        if prompt_pool:
            self.prompt = Prompt(
                length=64,
                embed_dim=chan,
                prompt_pool=True,
                pool_size=pool_size,
                top_k=top_k,
                batchwise_prompt=True,
                diversity_weight=diversity_weight,
                history_size=history_size
            )
            self.depth_transform = nn.Linear(384, chan)
            self.cross_attn = CrossAttention(chan, chan)

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, depth_feature=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = self.intro(inp)
        encs = [x]

        # Encoding
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Middle blocks with prompt
        x = self.middle_blks(x)
        
        if depth_feature is not None:
            # Ensure depth_feature is in shape [B, C, H, W]
            if depth_feature.dim() == 4:  # [B, C, H, W]
                depth_feature = depth_feature.view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, C]
            elif depth_feature.dim() == 3:  # [B, H*W, C] (already in compatible shape)
                pass
            else:
                raise ValueError(f"Unexpected depth_feature shape: {depth_feature.shape}")

            # Transform depth features
            depth_feature = self.depth_transform(depth_feature)  # [B, H*W, chan]
            
            # Get prompt embeddings and apply cross attention
            prompt_out = self.prompt(x.flatten(2).transpose(1, 2), depth_feature)
            prompt_embed = prompt_out.get('prompted_embedding', None)
            
            if prompt_embed is not None:
                x = self.cross_attn(x, prompt_embed)

        # Decoding
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x + encs[0])
        x = x[..., :H, :W]

        # Return both the restored image and prompt loss
        return x, 0.0  # The second value is the prompt loss, set to 0.0 if not using prompt loss


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
