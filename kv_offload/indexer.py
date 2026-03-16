"""分层索引构建：按key范数分层 + 余弦相似度索引"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HierarchicalIndex:
    """
    为每个attention head独立建立分层索引
    - 第一层：按key范数分桶
    - 第二层：每个桶内建立余弦相似度索引（支持快速搜索）
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        num_norm_buckets: int = 10,
        device: str = "cpu",
    ):
        """
        Args:
            num_layers: 模型层数
            num_kv_heads: KV heads数量
            num_norm_buckets: 范数分桶数量
            device: 索引存储设备（通常是cpu）
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_norm_buckets = num_norm_buckets
        self.device = device

        # 索引结构: {(layer_idx, head_idx, batch_idx): LayerHeadIndex}
        self.indices: Dict[Tuple[int, int, int], "LayerHeadIndex"] = {}

        logger.info(
            f"HierarchicalIndex initialized: {num_layers} layers, "
            f"{num_kv_heads} heads, {num_norm_buckets} norm buckets"
        )

    def build_index(
        self,
        layer_idx: int,
        head_idx: int,
        keys: torch.Tensor,
        token_indices: torch.Tensor,
        batch_idx: int = 0,
    ):
        """
        为指定layer和head建立索引
        Args:
            keys: [num_tokens, head_dim] 该head的所有keys（在CPU上）
            token_indices: [num_tokens] 对应的token位置
        """
        key = (layer_idx, head_idx, batch_idx)

        # 创建索引对象
        index = LayerHeadIndex(
            keys=keys,
            token_indices=token_indices,
            num_buckets=self.num_norm_buckets,
            device=self.device,
        )

        self.indices[key] = index

        logger.debug(
            f"Built index for layer {layer_idx}, head {head_idx}: "
            f"{len(token_indices)} tokens, {index.get_num_buckets()} buckets"
        )

    def search(
        self,
        layer_idx: int,
        head_idx: int,
        query: torch.Tensor,
        top_k: int,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """
        搜索与query最相似的top-k个keys
        Args:
            query: [head_dim] 单个query向量
            top_k: 返回top-k个最相似的keys
        Returns:
            token_indices: [top_k] 最相似的token位置索引
        """
        key = (layer_idx, head_idx, batch_idx)

        if key not in self.indices:
            # 没有索引，返回空
            return torch.empty(0, dtype=torch.long, device=self.device)

        return self.indices[key].search(query, top_k)

    def has_index(self, layer_idx: int, head_idx: int, batch_idx: int = 0) -> bool:
        """检查是否已建立索引"""
        return (layer_idx, head_idx, batch_idx) in self.indices

    def clear(self, layer_idx: Optional[int] = None, batch_idx: Optional[int] = None):
        """清空索引"""
        if layer_idx is not None and batch_idx is not None:
            keys_to_delete = [
                k
                for k in self.indices.keys()
                if k[0] == layer_idx and k[2] == batch_idx
            ]
            for k in keys_to_delete:
                del self.indices[k]
        elif layer_idx is not None:
            keys_to_delete = [k for k in self.indices.keys() if k[0] == layer_idx]
            for k in keys_to_delete:
                del self.indices[k]
        elif batch_idx is not None:
            keys_to_delete = [k for k in self.indices.keys() if k[2] == batch_idx]
            for k in keys_to_delete:
                del self.indices[k]
        else:
            self.indices.clear()


class LayerHeadIndex:
    """单个layer-head的分层索引"""

    def __init__(
        self,
        keys: torch.Tensor,
        token_indices: torch.Tensor,
        num_buckets: int = 10,
        device: str = "cpu",
    ):
        """
        Args:
            keys: [num_tokens, head_dim]
            token_indices: [num_tokens]
            num_buckets: 范数分桶数量
        """
        self.device = device
        self.num_buckets = num_buckets

        keys = keys.to(device)
        token_indices = token_indices.to(device)

        if keys.size(0) == 0:
            # 空索引
            self.buckets = []
            self.norm_ranges = []
            return

        # 计算每个key的L2范数
        norms = torch.norm(keys, p=2, dim=-1)  # [num_tokens]

        # 按范数分桶
        min_norm = norms.min().item()
        max_norm = norms.max().item()

        # 避免除零
        if max_norm - min_norm < 1e-6:
            # 所有key范数相同，放入一个桶
            self.buckets = [
                {
                    "keys": keys,  # [num_tokens, head_dim]
                    "token_indices": token_indices,  # [num_tokens]
                    "normalized_keys": F.normalize(
                        keys, p=2, dim=-1
                    ),  # 归一化用于余弦相似度
                }
            ]
            self.norm_ranges = [(min_norm, max_norm)]
        else:
            # 均匀分桶
            bucket_size = (max_norm - min_norm) / num_buckets
            self.buckets = []
            self.norm_ranges = []

            for i in range(num_buckets):
                bucket_min = min_norm + i * bucket_size
                bucket_max = (
                    min_norm + (i + 1) * bucket_size
                    if i < num_buckets - 1
                    else max_norm + 1e-6
                )

                # 找到在该范围内的keys
                mask = (norms >= bucket_min) & (norms < bucket_max)
                bucket_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

                if len(bucket_indices) > 0:
                    bucket_keys = keys[bucket_indices]  # [bucket_size, head_dim]
                    bucket_token_indices = token_indices[bucket_indices]

                    self.buckets.append(
                        {
                            "keys": bucket_keys,
                            "token_indices": bucket_token_indices,
                            "normalized_keys": F.normalize(bucket_keys, p=2, dim=-1),
                        }
                    )
                    self.norm_ranges.append((bucket_min, bucket_max))

    def search(self, query: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        搜索与query最相似的top-k个keys（余弦相似度）
        Args:
            query: [head_dim]
            top_k: 返回数量
        Returns:
            token_indices: [top_k] 最相似的token位置
        """
        if len(self.buckets) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        query = query.to(self.device)
        # 归一化query用于余弦相似度计算
        query_normalized = F.normalize(query.unsqueeze(0), p=2, dim=-1)  # [1, head_dim]

        # 在每个桶中搜索（可以优化为只搜索部分桶）
        all_scores = []
        all_token_indices = []

        for bucket in self.buckets:
            # 计算余弦相似度
            # bucket["normalized_keys"]: [bucket_size, head_dim]
            # query_normalized: [1, head_dim]
            scores = torch.matmul(
                query_normalized, bucket["normalized_keys"].T
            ).squeeze(0)  # [bucket_size]

            all_scores.append(scores)
            all_token_indices.append(bucket["token_indices"])

        # 合并所有桶的结果
        all_scores = torch.cat(all_scores, dim=0)  # [total_tokens]
        all_token_indices = torch.cat(all_token_indices, dim=0)  # [total_tokens]

        # 选择top-k
        top_k = min(top_k, len(all_scores))
        if top_k == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        top_k_scores, top_k_positions = torch.topk(all_scores, k=top_k, largest=True)
        top_k_token_indices = all_token_indices[top_k_positions]

        return top_k_token_indices

    def search_by_bucket(
        self, query: torch.Tensor, top_k_per_bucket: int
    ) -> torch.Tensor:
        """
        在每个桶中分别搜索top-k（更均匀的采样）
        Args:
            query: [head_dim]
            top_k_per_bucket: 每个桶返回的数量
        Returns:
            token_indices: [<=num_buckets * top_k_per_bucket]
        """
        if len(self.buckets) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        query = query.to(self.device)
        query_normalized = F.normalize(query.unsqueeze(0), p=2, dim=-1)

        all_token_indices = []

        for bucket in self.buckets:
            scores = torch.matmul(
                query_normalized, bucket["normalized_keys"].T
            ).squeeze(0)

            k = min(top_k_per_bucket, len(scores))
            if k > 0:
                _, top_positions = torch.topk(scores, k=k, largest=True)
                all_token_indices.append(bucket["token_indices"][top_positions])

        if len(all_token_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        return torch.cat(all_token_indices, dim=0)

    def get_num_buckets(self) -> int:
        """获取实际桶数量（可能小于num_buckets）"""
        return len(self.buckets)

    def get_bucket_sizes(self) -> List[int]:
        """获取每个桶的大小"""
        return [bucket["keys"].size(0) for bucket in self.buckets]
