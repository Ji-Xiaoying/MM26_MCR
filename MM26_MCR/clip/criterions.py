"""
定义损失函数/评估准则的文件
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def ce(sim_matrix):
    logpt = F.log_softmax(sim_matrix, dim=-1)
    logpt = torch.diag(logpt)
    return -logpt  # 计算对角线上的每个样本对应的交叉熵损失


class TotalLoss(nn.Module):
    def __init__(self, ):
        super(TotalLoss, self).__init__()

    def forward(self, i2t_sim_matrix, t2i_sim_matrix, subset=None):
        if subset == 'mbank':
            loss = (ce(i2t_sim_matrix) + ce(t2i_sim_matrix)) / 2
            return loss
        elif subset == 'batch':
            i2t_loss = ce(i2t_sim_matrix).mean()
            t2i_loss = ce(t2i_sim_matrix).mean()
            return i2t_loss, t2i_loss


class ReWeightLoss(nn.Module):
    def __init__(self, ):
        super(ReWeightLoss, self).__init__()

    def forward(self, i2t_sim_matrix, t2i_sim_matrix, weights):
        i2t_loss = ce(i2t_sim_matrix)
        t2i_loss = ce(t2i_sim_matrix)
        # 先对每个样本的损失进行加权, 然后取平均
        weighted_i2t_loss = (i2t_loss * weights).mean()
        weighted_t2i_loss = (t2i_loss * weights).mean()
        return weighted_i2t_loss, weighted_t2i_loss


class ReWeightLossv2(nn.Module):
    def __init__(self, ):
        super(ReWeightLossv2, self).__init__()

    def forward(self, i2t_sim_matrix, t2i_sim_matrix, i2t_weight, t2i_weight):
        i2t_loss = ce(i2t_sim_matrix)
        t2i_loss = ce(t2i_sim_matrix)
        # 先对每个样本的损失进行加权, 然后取平均
        weighted_i2t_loss = (i2t_loss * i2t_weight).mean()
        weighted_t2i_loss = (t2i_loss * t2i_weight).mean()
        return weighted_i2t_loss, weighted_t2i_loss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, i2t_sim_matrix, t2i_sim_matrix):
        i2t_loss = ce(i2t_sim_matrix)
        t2i_loss = ce(t2i_sim_matrix)
        return i2t_loss, t2i_loss


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, i2t_sim_matrix, t2i_sim_matrix, i2t_gt_matrix, t2i_gt_matrix):
        """
        :param i2t_sim_matrix: 图像到文本的相似度矩阵，形状为 [N, N]
        :param t2i_sim_matrix: 文本到图像的相似度矩阵，形状为 [N, N]
        :param i2t_gt_matrix: 图像到文本的ground truth矩阵，形状为 [N, N]
        :param t2i_gt_matrix: 文本到图像的ground truth矩阵，形状为 [N, N]
        :return: KL散度损失
        """
        # 归一化矩阵
        temperature = 0.5
        i2t_sim_matrix_normalized = F.softmax(i2t_sim_matrix, dim=1)
        t2i_sim_matrix_normalized = F.softmax(t2i_sim_matrix, dim=1)
        i2t_gt_matrix_normalized = F.softmax(i2t_gt_matrix / temperature, dim=1)  # 趋近于均匀分布, 减小temperature, 能扩大分布的差异
        t2i_gt_matrix_normalized = F.softmax(t2i_gt_matrix / temperature, dim=1)

        # 计算 KL 散度
        kl_i2t = self.kl_divergence(i2t_gt_matrix_normalized, i2t_sim_matrix_normalized)
        kl_t2i = self.kl_divergence(t2i_gt_matrix_normalized, t2i_sim_matrix_normalized)

        # 返回KL散度的总和
        return kl_i2t, kl_t2i

    def kl_divergence(self, p, q):
        """
        计算KL散度 D_KL(p || q) = sum(p * log(p / q))
        :param p: 目标概率分布
        :param q: 模型输出的实际概率分布
        :return: KL散度
        """
        epsilon = 1e-10  # 防止log(0)的错误
        p = p.to(torch.float64)
        q = q.to(torch.float64)
        p = torch.clamp(p, min=epsilon)  # 防止零概率
        q = torch.clamp(q, min=epsilon)  # 防止零概率

        return torch.sum(p * torch.log(p / q), dim=1).sum()  # 按行计算KL散度并求总和
        # bug:总是出现inf(无穷大)值
