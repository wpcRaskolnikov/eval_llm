"""KV检索：从CPU cache中检索相似的KV用于attention计算"""

import logging
from typing import Optional, Tuple

import torch

from .cpu_cache import CPUKVCache
from .indexer import HierarchicalIndex

logger = logging.getLogger(__name__)


class KVRetriever:
    """
    负责从CPU cache检索相似的KV
    - 接收GPU传来的query
    - 每个head独立检索top-k个最相似的keys
    - 合并所有heads检索到的token indices（去重）
    - 返回这些tokens的完整KV
    """

    def __init__(
        self,
        cpu_cache: CPUKVCache,
        indexer: HierarchicalIndex,
        top_k_per_head: int = 32,
    ):
        """
        Args:
            cpu_cache: CPU端KV cache
            indexer: 分层索引
            top_k_per_head: 每个head检索的token数量
        """
        self.cpu_cache = cpu_cache
        self.indexer = indexer
        self.top_k_per_head = top_k_per_head

        logger.info(f"KVRetriever initialized: top_k_per_head={top_k_per_head}")

    def retrieve(
        self,
        layer_idx: int,
        query: torch.Tensor,
        num_q_heads: int,
        batch_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        检索与query相似的KV
        Args:
            query: [batch_size, num_q_heads, 1, head_dim] decode时的query
            num_q_heads: query heads数量
        Returns:
            retrieved_keys: [batch_size, num_q_heads, num_retrieved, head_dim] 或 None
            retrieved_values: [batch_size, num_q_heads, num_retrieved, head_dim] 或 None
        """
        # 检查是否有CPU数据
        if not self.cpu_cache.has_data(layer_idx, batch_idx):
            return None, None

        batch_size = query.size(0)
        head_dim = query.size(3)
        num_kv_heads = self.cpu_cache.num_kv_heads

        # 收集所有heads检索到的token indices（去重）
        all_token_indices = set()

        # 为每个KV head检索（注意：如果是GQA，query heads > kv heads）
        n_rep = num_q_heads // num_kv_heads  # 每个KV head对应多少个Q head

        for kv_head_idx in range(num_kv_heads):
            # 对应的query head索引
            q_head_idx = kv_head_idx * n_rep

            # 获取该head的query
            q_vector = query[0, q_head_idx, 0, :]  # [head_dim]

            # 检索top-k
            if not self.indexer.has_index(layer_idx, kv_head_idx, batch_idx):
                continue

            token_indices = self.indexer.search(
                layer_idx=layer_idx,
                head_idx=kv_head_idx,
                query=q_vector,
                top_k=self.top_k_per_head,
                batch_idx=batch_idx,
            )

            # 添加到集合（自动去重）
            all_token_indices.update(token_indices.tolist())

        if len(all_token_indices) == 0:
            return None, None

        # 转换为tensor
        retrieve_indices = torch.tensor(
            sorted(list(all_token_indices)),
            dtype=torch.long,
            device="cpu",
        )

        # 从CPU cache获取这些tokens的KV
        keys_cpu, values_cpu = self.cpu_cache.get_subset(
            layer_idx, retrieve_indices, batch_idx
        )
        # keys_cpu: [num_kv_heads, num_retrieved, head_dim]

        if keys_cpu.size(1) == 0:
            return None, None

        # 转到GPU
        keys_gpu = keys_cpu.to(query.device)
        values_gpu = values_cpu.to(query.device)

        # 如果是GQA，扩展KV heads
        if num_q_heads != num_kv_heads:
            keys_gpu = keys_gpu.repeat(1, 1, 1)  # 先不repeat，在attention计算时repeat
            values_gpu = values_gpu.repeat(1, 1, 1)

        # 添加batch维度: [1, num_kv_heads, num_retrieved, head_dim]
        keys_gpu = keys_gpu.unsqueeze(0)
        values_gpu = values_gpu.unsqueeze(0)

        # GQA扩展
        if num_q_heads != num_kv_heads:
            keys_gpu = keys_gpu.repeat(1, n_rep, 1, 1)
            values_gpu = values_gpu.repeat(1, n_rep, 1, 1)

        logger.debug(
            f"Retrieved {keys_gpu.size(2)} tokens for layer {layer_idx} "
            f"(from {len(all_token_indices)} unique indices across all heads)"
        )

        return keys_gpu, values_gpu

    def retrieve_per_head(
        self,
        layer_idx: int,
        query: torch.Tensor,
        batch_idx: int = 0,
    ) -> list[Tuple[int, torch.Tensor]]:
        """
        每个head独立检索，返回每个head的检索结果
        Args:
            query: [batch_size, num_q_heads, 1, head_dim]
        Returns:
            List of (head_idx, token_indices)
        """
        results = []
        num_q_heads = query.size(1)
        num_kv_heads = self.cpu_cache.num_kv_heads
        n_rep = num_q_heads // num_kv_heads

        for kv_head_idx in range(num_kv_heads):
            q_head_idx = kv_head_idx * n_rep
            q_vector = query[0, q_head_idx, 0, :]

            if not self.indexer.has_index(layer_idx, kv_head_idx, batch_idx):
                continue

            token_indices = self.indexer.search(
                layer_idx=layer_idx,
                head_idx=kv_head_idx,
                query=q_vector,
                top_k=self.top_k_per_head,
                batch_idx=batch_idx,
            )

            results.append((kv_head_idx, token_indices))

        return results

    def update_top_k(self, new_top_k: int):
        """动态调整每个head检索的token数量"""
        self.top_k_per_head = new_top_k
        logger.info(f"Updated top_k_per_head to {new_top_k}")
