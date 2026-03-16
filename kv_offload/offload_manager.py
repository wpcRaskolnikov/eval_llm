import logging
from typing import Literal, Optional

import torch

from .cpu_cache import CPUKVCache
from .gpu_cache import GPUKVCache
from .hybrid_attention import hybrid_attention
from .indexer import HierarchicalIndex
from .retriever import KVRetriever

logger = logging.getLogger(__name__)


class HybridKVCacheManager:
    """
    混合KV Cache管理器：协调GPU和CPU cache，实现offload和检索

    使用流程：
    1. Prefill阶段：调用prefill()更新所有层的KV
    2. Offload：调用trigger_offload()将中间部分KV移到CPU并建立索引
    3. Decode阶段：调用decode()计算attention（自动协调CPU/GPU）
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch_size: int = 1,
        max_seq_len: int = 8192,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
        offload_ratio: float = 0.5,
        top_k_per_head: int = 32,
        num_norm_buckets: int = 10,
    ):
        """
        Args:
            num_layers: 模型层数
            num_kv_heads: KV attention heads数量
            head_dim: 每个head的维度
            max_batch_size: 最大batch size
            max_seq_len: 最大序列长度
            dtype: 数据类型
            device: GPU设备
            offload_ratio: offload的token比例（0-1）
            top_k_per_head: CPU检索时每个head检索的token数量
            num_norm_buckets: 范数分桶数量
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        self.offload_ratio = offload_ratio
        self.top_k_per_head = top_k_per_head

        # 初始化GPU cache
        self.gpu_cache = GPUKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

        # 初始化CPU cache
        self.cpu_cache = CPUKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            pin_memory=True,
        )

        # 初始化索引
        self.indexer = HierarchicalIndex(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            num_norm_buckets=num_norm_buckets,
            device="cpu",
        )

        # 初始化检索器
        self.retriever = KVRetriever(
            cpu_cache=self.cpu_cache,
            indexer=self.indexer,
            top_k_per_head=top_k_per_head,
        )

        # 标记是否已offload
        self.is_offloaded = False

        logger.info(
            f"HybridKVCacheManager initialized:\n"
            f"  Layers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}\n"
            f"  Max Seq Len: {max_seq_len}, Offload Ratio: {offload_ratio}\n"
            f"  Top-k per head: {top_k_per_head}, Norm Buckets: {num_norm_buckets}"
        )

    def prefill(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prefill阶段：将KV写入GPU cache
        Args:
            key_states: [batch_size, num_kv_heads, seq_len, head_dim]
            value_states: [batch_size, num_kv_heads, seq_len, head_dim]
        Returns:
            完整的key和value
        """
        return self.gpu_cache.update(layer_idx, key_states, value_states, batch_idx)

    def trigger_offload(
        self,
        strategy: Literal["middle", "random", "first"] = "middle",
        batch_idx: int = 0,
    ):
        """
        触发offload：将部分KV从GPU移到CPU并建立索引
        Args:
            strategy: offload策略
                - "middle": offload中间部分的tokens
                - "random": 随机offload
                - "first": offload前面的tokens
        """
        logger.info(f"Triggering offload with strategy: {strategy}")

        for layer_idx in range(self.num_layers):
            seq_len = self.gpu_cache.get_seq_len(batch_idx, layer_idx)

            if seq_len == 0:
                continue

            # 确定要offload的token位置
            num_offload = int(seq_len * self.offload_ratio)

            if num_offload == 0:
                continue

            if strategy == "middle":
                # Offload中间部分
                start_idx = (seq_len - num_offload) // 2
                offload_indices = torch.arange(
                    start_idx, start_idx + num_offload, dtype=torch.long
                )
            elif strategy == "random":
                # 随机选择
                offload_indices = torch.randperm(seq_len)[:num_offload].sort()[0]
            elif strategy == "first":
                # Offload前面的tokens
                offload_indices = torch.arange(num_offload, dtype=torch.long)
            else:
                raise ValueError(f"Unknown offload strategy: {strategy}")

            # 从GPU cache获取要offload的KV
            full_keys, full_values = self.gpu_cache.get(layer_idx, batch_idx)
            # [1, num_kv_heads, seq_len, head_dim]

            offload_keys = full_keys[
                0, :, offload_indices, :
            ]  # [heads, num_offload, dim]
            offload_values = full_values[0, :, offload_indices, :]

            # 存储到CPU cache
            self.cpu_cache.store(
                layer_idx=layer_idx,
                keys=offload_keys,
                values=offload_values,
                token_indices=offload_indices,
                batch_idx=batch_idx,
            )

            # 标记GPU cache中这些位置已offload
            self.gpu_cache.mark_offloaded(layer_idx, offload_indices, batch_idx)

            # 为每个head建立索引
            for head_idx in range(self.num_kv_heads):
                head_keys = offload_keys[head_idx, :, :]  # [num_offload, head_dim]
                self.indexer.build_index(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    keys=head_keys,
                    token_indices=offload_indices,
                    batch_idx=batch_idx,
                )

            logger.debug(
                f"Layer {layer_idx}: Offloaded {num_offload}/{seq_len} tokens "
                f"({num_offload / seq_len * 100:.1f}%)"
            )

        self.is_offloaded = True
        cpu_memory = self.cpu_cache.get_memory_usage_mb()
        logger.info(f"Offload completed. CPU cache size: {cpu_memory:.2f} MB")

    def decode(
        self,
        layer_idx: int,
        query: torch.Tensor,
        num_q_heads: int,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """
        Decode阶段：计算hybrid attention
        Args:
            query: [batch_size, num_q_heads, 1, head_dim]
            num_q_heads: query heads数量（可能和kv_heads不同，GQA）
        Returns:
            output: [batch_size, num_q_heads, 1, head_dim]
        """
        if not self.is_offloaded:
            # 没有offload，直接使用GPU cache计算
            output, _ = self.gpu_cache.compute_attention(
                layer_idx=layer_idx,
                query=query,
                batch_idx=batch_idx,
                return_lse=False,
            )
            return output

        # 1. GPU端计算（使用未offload的KV）
        o_gpu, lse_gpu = self.gpu_cache.compute_attention(
            layer_idx=layer_idx,
            query=query,
            batch_idx=batch_idx,
            return_lse=True,
        )

        # 2. CPU端检索和计算
        cpu_keys, cpu_values = self.retriever.retrieve(
            layer_idx=layer_idx,
            query=query,
            num_q_heads=num_q_heads,
            batch_idx=batch_idx,
        )

        # 3. 合并结果
        if cpu_keys is None or cpu_keys.size(2) == 0:
            # 没有检索到CPU KV，只使用GPU结果
            return o_gpu

        # 计算CPU部分的attention
        from .hybrid_attention import compute_cpu_attention, merge_attention_lse

        o_cpu, lse_cpu = compute_cpu_attention(
            query=query,
            keys=cpu_keys,
            values=cpu_values,
            head_dim=self.head_dim,
            return_lse=True,
        )

        # 合并GPU和CPU的结果
        o_merged, lse_merged = merge_attention_lse(o_gpu, lse_gpu, o_cpu, lse_cpu)

        return o_merged

    def update_decode(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_idx: int = 0,
    ):
        """
        Decode阶段更新：添加新生成的token的KV到GPU cache
        Args:
            key_states: [batch_size, num_kv_heads, 1, head_dim]
            value_states: [batch_size, num_kv_heads, 1, head_dim]
        """
        self.gpu_cache.update(layer_idx, key_states, value_states, batch_idx)

    def clear(self, batch_idx: Optional[int] = None):
        """清空所有cache"""
        self.gpu_cache.clear(batch_idx)
        self.cpu_cache.clear(batch_idx=batch_idx)
        self.indexer.clear(batch_idx=batch_idx)
        self.is_offloaded = False
        logger.info("Cache cleared")

    def get_statistics(self, batch_idx: int = 0) -> dict:
        """获取统计信息"""
        stats = {
            "is_offloaded": self.is_offloaded,
            "layers": {},
        }

        for layer_idx in range(self.num_layers):
            seq_len = self.gpu_cache.get_seq_len(batch_idx, layer_idx)
            num_gpu = self.gpu_cache.get_num_valid_tokens(layer_idx, batch_idx)
            num_cpu = self.cpu_cache.get_num_tokens(layer_idx, batch_idx)

            stats["layers"][layer_idx] = {
                "total_tokens": seq_len,
                "gpu_tokens": num_gpu,
                "cpu_tokens": num_cpu,
                "offload_ratio": num_cpu / seq_len if seq_len > 0 else 0,
            }

        stats["cpu_memory_mb"] = self.cpu_cache.get_memory_usage_mb()

        return stats

    def print_statistics(self, batch_idx: int = 0):
        """打印统计信息"""
        stats = self.get_statistics(batch_idx)

        print("\n" + "=" * 60)
        print("KV Cache Statistics")
        print("=" * 60)
        print(f"Offload Status: {'Enabled' if stats['is_offloaded'] else 'Disabled'}")
        print(f"CPU Memory Usage: {stats['cpu_memory_mb']:.2f} MB")
        print("\nPer-Layer Breakdown:")
        print(f"{'Layer':<8} {'Total':<8} {'GPU':<8} {'CPU':<8} {'Offload%':<10}")
        print("-" * 60)

        for layer_idx, layer_stats in stats["layers"].items():
            print(
                f"{layer_idx:<8} "
                f"{layer_stats['total_tokens']:<8} "
                f"{layer_stats['gpu_tokens']:<8} "
                f"{layer_stats['cpu_tokens']:<8} "
                f"{layer_stats['offload_ratio'] * 100:<10.1f}"
            )

        print("=" * 60 + "\n")
