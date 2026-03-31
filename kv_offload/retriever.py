"""KV检索：从CPU cache中检索相似的KV用于attention计算"""

import logging
import math
from typing import Optional, Tuple

import torch

from .cpu_cache import CPUKVCache
from .indexer import HierarchicalIndex

# log2(e)：用于将自然对数 lse 转换为 log2-base lse，与 flashinfer 对齐
_LOG2E = math.log2(math.e)

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
        if not self.cpu_cache.has_data(batch_idx):
            return None, None

        _batch_size = query.size(0)
        _head_dim = query.size(3)
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
        keys_cpu, values_cpu = self.cpu_cache.get_by_indices(
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

    def retrieve_and_compute(
        self,
        layer_idx: int,
        query: torch.Tensor,
        num_q_heads: int,
        device: torch.device,
        head_dim: int,
        batch_idx: int = 0,
        use_log2_lse: bool = True,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        所有 KV head 检索到的 token 取并集，每个 head 在并集上计算 attention。

        Args:
            query: [1, num_q_heads, 1, head_dim]  在GPU上
            num_q_heads: query head数量（GQA时 > num_kv_heads）
            device: 计算设备（GPU）
            head_dim: head维度
            use_log2_lse: 若 True（默认），返回的 lse 使用 log2 base，与 flashinfer
                          的 merge_state 保持一致；若 False，返回自然对数 base 的 lse。

        Returns:
            (o_cpu, lse_cpu) 若 CPU cache 有数据：
                o_cpu:   [1, 1, num_q_heads, head_dim]  在GPU上
                lse_cpu: [1, num_q_heads, 1]             float32，在GPU上
            None 若 CPU cache 无数据
        """
        if not self.cpu_cache.has_data(batch_idx):
            return None

        num_kv_heads = self.cpu_cache.num_kv_heads
        n_rep = num_q_heads // num_kv_heads

        scale = 1.0 / (head_dim**0.5)

        # ---- 第一遍：所有 kv_head 检索结果取并集 ----
        all_token_positions: set = set()
        for kv_head_idx in range(num_kv_heads):
            if not self.indexer.has_index(layer_idx, kv_head_idx, batch_idx):
                continue
            q_head_start = kv_head_idx * n_rep
            q_search = query[0, q_head_start, 0, :].cpu()
            top_k_positions = self.indexer.search(
                layer_idx=layer_idx,
                head_idx=kv_head_idx,
                query=q_search,
                top_k=self.top_k_per_head,
                batch_idx=batch_idx,
            )
            all_token_positions.update(top_k_positions.tolist())

        if len(all_token_positions) == 0:
            return None

        retrieve_indices = torch.tensor(sorted(all_token_positions), dtype=torch.long)
        keys_union, values_union = self.cpu_cache.get_by_indices(
            layer_idx, retrieve_indices, batch_idx
        )
        keys_union = keys_union.to(dtype=query.dtype, device=device)
        values_union = values_union.to(dtype=query.dtype, device=device)

        # GQA 扩展: [num_kv_heads, ...] -> [num_q_heads, ...]
        if n_rep > 1:
            keys_union = keys_union.repeat_interleave(n_rep, dim=0)
            values_union = values_union.repeat_interleave(n_rep, dim=0)

        # q: [num_q_heads, 1, head_dim]
        q = query[0, :, 0, :].unsqueeze(1)

        # scores: [num_q_heads, 1, num_union] -> [num_q_heads, num_union]
        scores = torch.bmm(q, keys_union.transpose(1, 2)).squeeze(1) * scale

        max_s = scores.max(dim=-1, keepdim=True).values  # [num_q_heads, 1]
        exp_shifted = torch.exp(scores - max_s)  # [num_q_heads, num_union]
        sum_exp = exp_shifted.sum(dim=-1, keepdim=True)  # [num_q_heads, 1]
        lse = (
            (max_s + torch.log(sum_exp)).squeeze(-1).to(torch.float32)
        )  # [num_q_heads]
        if use_log2_lse:
            lse = lse * _LOG2E  # 转换为 log2 base，与 flashinfer 对齐

        attn_w = exp_shifted / sum_exp  # [num_q_heads, num_union]
        # out: [num_q_heads, 1, num_union] x [num_q_heads, num_union, head_dim] -> [num_q_heads, head_dim]
        out = torch.bmm(attn_w.unsqueeze(1), values_union).squeeze(1)

        # o_cpu: [1, 1, num_q_heads, head_dim]
        o_cpu = out.unsqueeze(0).unsqueeze(1)
        # lse_cpu: [1, num_q_heads, 1]
        lse_cpu = lse.unsqueeze(0).unsqueeze(-1)

        return o_cpu, lse_cpu

    def update_top_k(self, new_top_k: int):
        """动态调整每个head检索的token数量"""
        self.top_k_per_head = new_top_k
        logger.info(f"Updated top_k_per_head to {new_top_k}")
