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

    def retrieve_and_compute(
        self,
        layer_idx: int,
        query: torch.Tensor,
        num_q_heads: int,
        batch_idx: int = 0,
        device: torch.device = None,
        head_dim: int = 64,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        每个KV head独立检索自己的top-k token，并计算对应q_head组的attention。

        每个kv_head用组内第一个q_head的向量搜索索引，找到top-k个最相似的token，
        然后对该组所有q_head分别计算 scaled dot-product attention，
        最终拼接成完整的 cpu attention 输出。

        Args:
            query: [1, num_q_heads, 1, head_dim]  在GPU上
            num_q_heads: query head数量（GQA时 > num_kv_heads）
            device: 计算设备（GPU）
            head_dim: head维度

        Returns:
            o_cpu:   [1, num_q_heads, 1, head_dim]  在GPU上
            lse_cpu: [1, num_q_heads, 1]             float32，在GPU上
            两者均为 None 若 CPU cache 无数据
        """
        if not self.cpu_cache.has_data(layer_idx, batch_idx):
            return None, None

        num_kv_heads = self.cpu_cache.num_kv_heads
        n_rep = num_q_heads // num_kv_heads
        if device is None:
            device = query.device

        # 一次性取出该层所有CPU上的KV
        # keys_all: [num_kv_heads, num_stored, head_dim]  on CPU
        # stored_indices: [num_stored]  token原始位置
        keys_all, values_all, stored_indices = self.cpu_cache.get(layer_idx, batch_idx)

        if keys_all.size(1) == 0:
            return None, None

        num_stored = stored_indices.size(0)
        scale = 1.0 / (head_dim**0.5)

        # 建立 token_position -> storage_position 映射，避免重复扫描
        # stored_indices 在CPU上，转成 python dict 供快速查询
        pos_map: dict = {int(stored_indices[i].item()): i for i in range(num_stored)}

        outputs = []  # 每个 q_head 对应一个 [head_dim] tensor
        lses: list = []  # 每个 q_head 对应一个标量 tensor (float32)

        for kv_head_idx in range(num_kv_heads):
            q_head_start = kv_head_idx * n_rep

            # 用该 kv_head 组内第一个 q_head 的向量做索引检索
            q_search = query[0, q_head_start, 0, :].cpu()  # [head_dim]

            # ---- 检索阶段（在CPU上完成） ----
            if not self.indexer.has_index(layer_idx, kv_head_idx, batch_idx):
                # 该head无索引：填零输出 + -inf LSE
                zero_out = torch.zeros(head_dim, dtype=query.dtype, device=device)
                inf_lse = torch.full(
                    (), float("-inf"), dtype=torch.float32, device=device
                )
                for _ in range(n_rep):
                    outputs.append(zero_out)
                    lses.append(inf_lse)
                continue

            top_k_token_positions = self.indexer.search(
                layer_idx=layer_idx,
                head_idx=kv_head_idx,
                query=q_search,
                top_k=self.top_k_per_head,
                batch_idx=batch_idx,
            )  # [top_k] —— 原始token位置，在CPU上

            if len(top_k_token_positions) == 0:
                zero_out = torch.zeros(head_dim, dtype=query.dtype, device=device)
                inf_lse = torch.full(
                    (), float("-inf"), dtype=torch.float32, device=device
                )
                for _ in range(n_rep):
                    outputs.append(zero_out)
                    lses.append(inf_lse)
                continue

            # 将 token 位置映射到存储下标
            storage_pos = [
                pos_map[int(t.item())]
                for t in top_k_token_positions
                if int(t.item()) in pos_map
            ]
            if len(storage_pos) == 0:
                zero_out = torch.zeros(head_dim, dtype=query.dtype, device=device)
                inf_lse = torch.full(
                    (), float("-inf"), dtype=torch.float32, device=device
                )
                for _ in range(n_rep):
                    outputs.append(zero_out)
                    lses.append(inf_lse)
                continue

            storage_pos_t = torch.tensor(storage_pos, dtype=torch.long)

            # 取出该 kv_head 检索到的 K/V，传到GPU
            # k_head: [num_found, head_dim]
            k_head = keys_all[kv_head_idx, storage_pos_t, :].to(
                dtype=query.dtype, device=device
            )
            v_head = values_all[kv_head_idx, storage_pos_t, :].to(
                dtype=query.dtype, device=device
            )

            # ---- 计算阶段（在GPU上完成） ----
            # 对该组内每个 q_head 单独计算 attention
            for rep_idx in range(n_rep):
                q_head_idx = q_head_start + rep_idx
                q_vec = query[0, q_head_idx, 0, :]  # [head_dim]

                # scores: [num_found]
                scores = torch.mv(k_head, q_vec) * scale  # k_head @ q_vec^T

                # 数值稳定的 LSE
                max_s = scores.max()
                exp_shifted = torch.exp(scores - max_s)
                sum_exp = exp_shifted.sum()
                lse_h = (max_s + torch.log(sum_exp)).to(torch.float32)

                # softmax + weighted sum
                attn_w = exp_shifted / sum_exp  # [num_found]
                out = torch.mv(v_head.T, attn_w)  # [head_dim]  = v_head^T @ attn_w

                outputs.append(out)
                lses.append(lse_h)

        # 拼装输出
        # o_cpu: [1, num_q_heads, 1, head_dim]
        o_cpu = torch.stack(outputs, dim=0).unsqueeze(0).unsqueeze(2)

        # lse_cpu: [1, num_q_heads, 1]
        lse_cpu = torch.stack(lses, dim=0).unsqueeze(0).unsqueeze(-1)

        logger.debug(
            f"Layer {layer_idx}: per-head CPU attention computed, "
            f"o_cpu shape={o_cpu.shape}"
        )

        return o_cpu, lse_cpu

    def update_top_k(self, new_top_k: int):
        """动态调整每个head检索的token数量"""
        self.top_k_per_head = new_top_k
        logger.info(f"Updated top_k_per_head to {new_top_k}")
