import logging
from typing import Optional, Tuple

import torch

from .cpu_cache import CPUKVCache
from .indexer import HierarchicalIndex

logger = logging.getLogger(__name__)


class KVRetriever:
    """
    Retrieves KV pairs from CPU cache for a given query.
    - Each KV head independently retrieves top-k most similar keys
    - Token indices from all heads are merged (deduplicated)
    - Returns the corresponding KV tensors
    """

    def __init__(
        self,
        cpu_cache: CPUKVCache,
        indexer: HierarchicalIndex,
        top_k_per_head: int = 32,
    ):
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
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: [1, num_q_heads, 1, head_dim]
        Returns:
            (keys, values): [1, num_q_heads, num_retrieved, head_dim] or None
        """
        if not self.cpu_cache.has_data(batch_idx):
            return None

        if not self.indexer.has_index(layer_idx, batch_idx):
            return None

        num_kv_heads = self.cpu_cache.num_kv_heads
        n_rep = num_q_heads // num_kv_heads

        all_token_indices: set = set()
        for kv_head_idx in range(num_kv_heads):
            # # Only use first q_head per group, misses other q_heads' attention patterns
            # q_vector = query[0, kv_head_idx * n_rep, 0, :].cpu()
            # token_indices = self.indexer.search(
            #     layer_idx=layer_idx,
            #     head_idx=kv_head_idx,
            #     query=q_vector,
            #     top_k=self.top_k_per_head,
            #     batch_idx=batch_idx,
            # )
            # all_token_indices.update(token_indices.tolist())
            for rep_idx in range(n_rep):
                q_vector = query[0, kv_head_idx * n_rep + rep_idx, 0, :].cpu()
                token_indices = self.indexer.search(
                    layer_idx=layer_idx,
                    head_idx=kv_head_idx,
                    query=q_vector,
                    top_k=self.top_k_per_head,
                    batch_idx=batch_idx,
                )
                all_token_indices.update(token_indices.tolist())

        if not all_token_indices:
            return None

        retrieve_indices = torch.tensor(sorted(all_token_indices), dtype=torch.long)
        keys, values = self.cpu_cache.get_by_indices(
            layer_idx, retrieve_indices, batch_idx
        )  # [num_kv_heads, num_retrieved, head_dim]

        keys = keys.to(dtype=query.dtype, device=query.device)
        values = values.to(dtype=query.dtype, device=query.device)

        # [num_kv_heads, ...] -> [num_q_heads, ...]
        if n_rep > 1:
            keys = keys.repeat_interleave(n_rep, dim=0)
            values = values.repeat_interleave(n_rep, dim=0)

        # [1, num_q_heads, num_retrieved, head_dim]
        return keys.unsqueeze(0), values.unsqueeze(0)

    def update_top_k(self, new_top_k: int):
        self.top_k_per_head = new_top_k
        logger.info(f"Updated top_k_per_head to {new_top_k}")
