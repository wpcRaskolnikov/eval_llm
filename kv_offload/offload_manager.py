import logging
import math
from typing import Literal, Optional

import flashinfer
import torch

from .cpu_cache import CPUKVCache
from .gpu_cache import GPUKVCache
from .indexer import HierarchicalIndex
from .retriever import KVRetriever

# log2(e): converts natural log LSE to log2 base, aligning with flashinfer.merge_state
_LOG2E = math.log2(math.e)

logger = logging.getLogger(__name__)


class HybridKVCacheManager:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch_size: int = 1,
        max_seq_len: int = 8192,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
        offload_ratio: float = 0.5,
        top_k_per_head: int = 8,
        num_norm_buckets: int = 10,
        hnsw_M: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        self.offload_ratio = offload_ratio
        self.top_k_per_head = top_k_per_head

        self.gpu_cache = GPUKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

        self.cpu_cache = CPUKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            pin_memory=True,
        )

        self.indexer = HierarchicalIndex(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            num_norm_buckets=num_norm_buckets,
            M=hnsw_M,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
            device="cpu",
        )

        self.retriever = KVRetriever(
            cpu_cache=self.cpu_cache,
            indexer=self.indexer,
            top_k_per_head=top_k_per_head,
        )

        self.is_offloaded = False

        logger.info(
            f"HybridKVCacheManager initialized:\n"
            f"  Layers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}\n"
            f"  Max Seq Len: {max_seq_len}, Offload Ratio: {offload_ratio}\n"
            f"  Top-k per head: {top_k_per_head}, HNSW M={hnsw_M}, ef_search={hnsw_ef_search}, num_norm_buckets={num_norm_buckets}"
        )

    def prefill(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, num_q_heads, seq_len, head_dim]
            key_states: [batch_size, num_kv_heads, seq_len, head_dim]
            value_states: [batch_size, num_kv_heads, seq_len, head_dim]
        Returns:
            attn_output: [1, seq_len, num_q_heads, head_dim]
        """
        self.gpu_cache.update(layer_idx, key_states, value_states, batch_idx)
        attn_output, _ = self.gpu_cache.compute_attention(
            layer_idx=layer_idx,
            query=query,
            is_prefill=True,
            batch_idx=batch_idx,
        )
        return attn_output

    def trigger_offload(
        self,
        strategy: Literal["middle", "random", "first"] = "middle",
        batch_idx: int = 0,
    ):
        logger.info(f"Triggering offload with strategy: {strategy}")

        for layer_idx in range(self.num_layers):
            seq_len = self.gpu_cache.get_seq_len(layer_idx, batch_idx)

            if seq_len == 0:
                continue

            num_offload = int(seq_len * self.offload_ratio)

            if num_offload == 0:
                continue

            if strategy == "middle":
                start_idx = (seq_len - num_offload) // 2
                offload_indices = torch.arange(
                    start_idx, start_idx + num_offload, dtype=torch.long
                )
            elif strategy == "random":
                offload_indices = torch.randperm(seq_len)[:num_offload].sort()[0]
            elif strategy == "first":
                offload_indices = torch.arange(num_offload, dtype=torch.long)
            else:
                raise ValueError(f"Unknown offload strategy: {strategy}")

            full_keys, full_values = self.gpu_cache.get(layer_idx, batch_idx)
            # [1, num_kv_heads, seq_len, head_dim]

            offload_keys = full_keys[
                0, :, offload_indices, :
            ]  # [heads, num_offload, dim]
            offload_values = full_values[0, :, offload_indices, :]
            self.cpu_cache.store(
                layer_idx=layer_idx,
                keys=offload_keys,
                values=offload_values,
                token_indices=offload_indices,
                batch_idx=batch_idx,
            )
            self.gpu_cache.mark_offloaded(layer_idx, offload_indices, batch_idx)

            for head_idx in range(self.num_kv_heads):
                head_keys = offload_keys[head_idx, :, :]  # [num_offload, head_dim]
                self.indexer.build_index(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    keys=head_keys,
                    token_indices=offload_indices,
                    batch_idx=batch_idx,
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
        Args:
            query: [batch_size, num_q_heads, 1, head_dim]
            num_q_heads: query heads数量（GQA时 > num_kv_heads）
        Returns:
            output: [batch_size, num_q_heads, 1, head_dim]
        """
        if not self.is_offloaded:
            output, _ = self.gpu_cache.compute_attention(
                layer_idx=layer_idx,
                query=query,
                is_prefill=False,
                batch_idx=batch_idx,
                return_lse=False,
            )
            return output

        o_gpu, lse_gpu = self.gpu_cache.compute_attention(
            layer_idx=layer_idx,
            query=query,
            is_prefill=False,
            batch_idx=batch_idx,
            return_lse=True,
        )
        # o_gpu:   [1, 1, num_q_heads, head_dim]
        # lse_gpu: [1, num_q_heads, 1]

        retrieved = self.retriever.retrieve(
            layer_idx=layer_idx,
            query=query,
            num_q_heads=num_q_heads,
            batch_idx=batch_idx,
        )

        if retrieved is None:
            return o_gpu

        keys, values = retrieved

        # keys/values: [1, num_q_heads, num_retrieved, head_dim]
        scale = 1.0 / (self.head_dim**0.5)
        q = query[0, :, 0, :]  # [num_q_heads, head_dim]
        k = keys[0]  # [num_q_heads, num_retrieved, head_dim]
        v = values[0]  # [num_q_heads, num_retrieved, head_dim]

        scores = (
            torch.einsum("hd,hnd->hn", q, k) * scale
        )  # [num_q_heads, num_retrieved]
        max_s = scores.max(dim=-1, keepdim=True).values  # [num_q_heads, 1]
        exp_shifted = torch.exp(scores - max_s)
        sum_exp = exp_shifted.sum(dim=-1, keepdim=True)  # [num_q_heads, 1]
        attn_w = exp_shifted / sum_exp  # [num_q_heads, num_retrieved]

        o_cpu = torch.einsum("hn,hnd->hd", attn_w, v).unsqueeze(0)
        lse_cpu = (
            (max_s + torch.log(sum_exp)).squeeze(-1).to(torch.float32) * _LOG2E
        ).unsqueeze(0)
        # o_cpu: [1, num_q_heads, head_dim]
        # lse_cpu: [1, num_q_heads]

        o_merged, _ = flashinfer.merge_state(
            o_gpu.squeeze(1),  # [1, num_q_heads, head_dim]
            lse_gpu.squeeze(2).float(),  # [1, num_q_heads]
            o_cpu,
            lse_cpu,
        )
        return o_merged.unsqueeze(1)

    def append_kv(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_idx: int = 0,
    ):
        """
        Args:
            key_states: [batch_size, num_kv_heads, 1, head_dim]
            value_states: [batch_size, num_kv_heads, 1, head_dim]
        """
        self.gpu_cache.update(layer_idx, key_states, value_states, batch_idx)

    def clear(self, batch_idx: Optional[int] = None):
        self.gpu_cache.clear(batch_idx)
        self.cpu_cache.clear(batch_idx)
        self.indexer.clear(batch_idx)
        self.is_offloaded = False
        logger.info("Cache cleared")

    def get_seq_len(self, layer_idx: int, batch_idx: int = 0) -> int:
        return self.gpu_cache.get_seq_len(batch_idx=batch_idx, layer_idx=layer_idx)

    def get_statistics(self, batch_idx: int = 0) -> dict:
        stats = {
            "is_offloaded": self.is_offloaded,
            "layers": {},
        }

        for layer_idx in range(self.num_layers):
            seq_len = self.gpu_cache.get_seq_len(layer_idx, batch_idx)
            num_gpu = self.gpu_cache.get_num_valid_tokens(layer_idx, batch_idx)
            num_cpu = self.cpu_cache.get_num_tokens(batch_idx)

            stats["layers"][layer_idx] = {
                "total_tokens": seq_len,
                "gpu_tokens": num_gpu,
                "cpu_tokens": num_cpu,
                "offload_ratio": num_cpu / seq_len if seq_len > 0 else 0,
            }

        stats["cpu_memory_mb"] = self.cpu_cache.get_memory_usage_mb()

        return stats

    def print_statistics(self, batch_idx: int = 0):
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
