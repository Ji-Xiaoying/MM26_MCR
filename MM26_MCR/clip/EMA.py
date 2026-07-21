from collections import deque
import numpy as np
import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}  # 维护EMA版本的模型参数
        self.backup = {}  # 保存原始模型参数,确保模型在apply_shadow()更新参数后仍可恢复原始参数

        for name, param in self.model.named_parameters():
            if param.requires_grad:  # 遍历模型的所有可训练参数
                self.shadow[name] = param.data.clone()  # 深拷贝模型的参数值

    def update(self):  # 更新EMA模型参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        # 应用EMA模型参数到model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()  # 保存原始模型的权重
                param.data = self.shadow[name]

    def restore(self):
        # 恢复原始模型权重
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class SampleQueue:
    def __init__(self, max_size=500, strategy='confidence'):
        self.queue = deque(maxlen=max_size)
        self.strategy = strategy

    def update(self, samples, confidences=None):
        if self.strategy == 'FIFO':  # 先进先出
            self.queue.extend(samples)

        elif self.strategy == 'confidence' and confidences is not None:
            sorted_samples = [x for _, x in sorted(zip(confidences, samples), reverse=True)]
            self.queue.extend(sorted_samples[:len(samples) // 2])  # 仅保留高置信度的一半样本

        elif self.strategy == 'confidence_diversity' and confidences is not None:
            sorted_samples = [x for _, x in sorted(zip(confidences, samples), reverse=True)]
            diverse_samples = self._select_diverse_samples(sorted_samples)
            self.queue.extend(diverse_samples)

    def _select_diverse_samples(self, samples, threshold=0.8):
        selected = []
        for s in samples:
            if all(np.linalg.norm(s - x) > threshold for x in selected):  # 计算欧氏距离，保证队列中的样本多样性
                selected.append(s)
            if len(selected) >= len(samples) // 2:
                break
        return selected

    def get_samples(self):
        # # 将每个张量移动到 CPU 并转换为 numpy 数组
        # numpy_list = [tensor.detach().cpu().numpy() for tensor in self.queue]
        #
        # # 将所有 numpy 数组堆叠在一起
        # if numpy_list == []:
        #     return None
        # numpy_array = np.stack(numpy_list)
        # return numpy_array
        # 将队列中的张量堆叠在一起
        if not self.queue:
            return None
        tensor_list = list(self.queue)
        tensor_array = torch.stack(tensor_list)
        return tensor_array
