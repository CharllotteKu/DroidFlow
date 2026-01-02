import torch
import torch.nn as nn
import numpy as np

# -----------------------------------------------------------------
# 基础组件：Affine Coupling Layer 的子网络 (保持不变)
# -----------------------------------------------------------------
class Nett(nn.Module):
    def __init__(self, dim, h):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * h)
        self.fc2 = nn.Linear(dim * h, dim * h)
        self.fc3 = nn.Linear(dim * h, dim)
        self.activation = nn.ReLU() # 使用 ReLU

    def forward(self, x):
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(x)))))

class Nets(nn.Module):
    def __init__(self, dim, h):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * h)
        self.fc2 = nn.Linear(dim * h, dim * h)
        self.fc3 = nn.Linear(dim * h, dim)
        self.activation = nn.Tanh() # Scale 使用 Tanh 保持数值稳定

    def forward(self, x):
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(x)))))

# -----------------------------------------------------------------
# 核心模型：AdvancedFlow
# -----------------------------------------------------------------
class AdvancedFlow(nn.Module):
    def __init__(self, dim, h=1, num_blocks=5, layers_per_block=3):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block

        # 构建掩码 (Masks)
        self.masks = nn.ParameterList()
        # 构建耦合层 (S和T网络)
        self.nett = nn.ModuleList()
        self.nets = nn.ModuleList()
        # 构建可逆1x1卷积 (此处简化为全连接矩阵)
        self.perm_weights = nn.ParameterList()
        
        # 基础分布参数 (用于计算 likelihood)
        self.register_buffer('prior_mean', torch.zeros(dim))
        self.register_buffer('prior_log_sd', torch.zeros(dim))

        count = 0
        for b in range(num_blocks):
            # 每个 Block 开始前做一个随机置换 (Invertible 1x1 简化版)
            # 使用 LU 分解或简单的正交矩阵更好，这里用简单的随机矩阵初始化
            w_init = torch.linalg.qr(torch.randn(dim, dim))[0]
            self.perm_weights.append(nn.Parameter(w_init))
            
            for l in range(layers_per_block):
                # 棋盘掩码或交替掩码
                mask = torch.zeros(dim)
                if count % 2 == 0:
                    mask[::2] = 1
                else:
                    mask[1::2] = 1
                self.masks.append(nn.Parameter(mask, requires_grad=False))
                
                self.nett.append(Nett(dim, h))
                self.nets.append(Nets(dim, h))
                count += 1

    def _transform(self, x):
        """
        核心流变换逻辑：从 X -> Z
        """
        # --- 修复点：初始化 sldj 为与 batch size 匹配的零向量 ---
        batch_size = x.shape[0]
        sldj = torch.zeros(batch_size, device=x.device) 
        
        z = x
        
        layer_idx = 0
        for b in range(self.num_blocks):
            # 1. 线性层 (Permutation / Invertible 1x1)
            # z = z @ W
            W = self.perm_weights[b]
            z = torch.matmul(z, W)
            
            # Jacobian = det(W), LogJac = log|det(W)|
            # torch.slogdet 返回 (sign, logabsdet)
            # 我们只需要 logabsdet
            w_log_det = torch.slogdet(W)[1]
            
            # w_log_det 是标量，广播加到 sldj
            sldj = sldj + w_log_det
            
            # 2. 非线性耦合层 (Affine Coupling)
            for l in range(self.layers_per_block):
                mask = self.masks[layer_idx]
                t_net = self.nett[layer_idx]
                s_net = self.nets[layer_idx]
                
                z_masked = z * mask
                s = s_net(z_masked)
                t = t_net(z_masked)
                
                # Affine 变换: z' = mask*z + (1-mask)*(z * exp(s) + t)
                z = z_masked + (1 - mask) * (z * torch.exp(s) + t)
                
                # 更新 log determinant
                # 只有被变换的部分 (1-mask) 贡献了 s
                # sum((1-mask)*s, dim=1) 结果是 [batch_size]
                # sldj 也是 [batch_size]，现在可以直接相加
                log_det_coupling = torch.sum((1 - mask) * s, dim=1)
                sldj = sldj + log_det_coupling
                
                layer_idx += 1
                
        return z, sldj

    def forward(self, x):
        """
        前向传播：计算 z 和 loss 所需组件
        """
        z, sldj = self._transform(x)
        
        # 为了匹配 FlowConLoss 的接口: (z, sldj, means, log_sds)
        # 我们使用标准正态分布作为 Prior，所以 means=0, log_sds=0
        batch_size = x.shape[0]
        means = self.prior_mean.expand(batch_size, -1)
        log_sds = self.prior_log_sd.expand(batch_size, -1)
        
        return z, means, log_sds, sldj

    def log_prob(self, x):
        """
        计算 Log Likelihood P(x)
        用于评估阶段
        """
        z, means, log_sds, sldj = self.forward(x)
        
        # Log P(z) ~ N(0, 1)
        # -0.5 * (z^2 + log(2pi))
        log_p_z = -0.5 * (z ** 2 + np.log(2 * np.pi))
        log_p_z = torch.sum(log_p_z, dim=1)
        
        # Log P(x) = Log P(z) + Log |det J|
        return log_p_z + sldj