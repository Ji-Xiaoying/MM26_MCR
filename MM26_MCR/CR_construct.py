import torch
import torch.nn.functional as F


def multi_phase_CR(image_features, text_features, lambda_reg, topk_list=(128, 64, 32, 16, 8, 4, -1)):
    image_features = F.normalize(image_features.float(), dim=1)
    text_features = F.normalize(text_features.float(), dim=1)

    B, D = image_features.shape
    device = image_features.device

    # ===== 初始化 =====
    # 构建初始字典：每个图像都用整个 Batch 的文本作为候选集
    dict_features = text_features.unsqueeze(0).expand(B, -1, -1)  # (B, B, D)

    # 记录字典中每个文本在原始 Batch 中的索引
    dict_indices = torch.arange(text_features.shape[0], device=device) \
        .unsqueeze(0).expand(B, -1)  # (B, B)

    final_score = torch.zeros(B, device=device)
    stage_scores_list = []

    # ===== 迭代多阶段 =====
    for stage_id, k in enumerate(topk_list):
        M_cur = dict_features.shape[1]

        # 计算协同表示系数
        G = torch.bmm(dict_features, dict_features.transpose(1, 2))
        I = torch.eye(M_cur, device=device).unsqueeze(0)
        inv = torch.inverse(G + lambda_reg * I)

        img = image_features.unsqueeze(-1)
        S = torch.bmm(dict_features, img)
        C = torch.bmm(inv, S).squeeze(-1)

        C_norm = torch.norm(C, p=1, dim=1, keepdim=True) + 1e-8
        C = C / C_norm

        target_idx = torch.arange(B, device=device).unsqueeze(1)  # [0,B-1]
        mask = (dict_indices == target_idx)  # 只提取主对角线的系数

        stage_score = (C * mask.float()).sum(dim=1)
        final_score += F.relu(stage_score)  # 累加置信度

        stage_scores_list.append(F.relu(stage_score).unsqueeze(1))

        # ---------- top-k shrink ----------
        if k != -1 and k < M_cur:
            # 选出系数最大的 k 个文本保留到下一轮
            _, topk_idx = torch.topk(C, k=k, dim=1)

            dict_features = torch.gather(
                dict_features, 1,
                topk_idx.unsqueeze(-1).expand(-1, -1, D)
            )

            dict_indices = torch.gather(
                dict_indices, 1, topk_idx
            )

    all_stage_scores = torch.cat(stage_scores_list, dim=1)

    return final_score, all_stage_scores
