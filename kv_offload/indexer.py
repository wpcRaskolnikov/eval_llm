import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BucketEntry:
    hnsw: hnswlib.Index
    num_tokens: int


class LayerHeadIndex:
    def __init__(
        self,
        keys: torch.Tensor,
        token_indices: torch.Tensor,
        num_buckets: int = 10,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        """
        Args:
            keys: [num_tokens, head_dim]
            token_indices: [num_tokens]
            num_buckets: number of norm-based buckets
            M/ef_construction/ef_search: HNSW parameters
        """
        self.ef_search = ef_search
        self.buckets: List[BucketEntry] = []
        self.norm_ranges: List[Tuple[float, float]] = []

        head_dim = keys.shape[1]
        keys_cpu = keys.cpu()
        token_indices_cpu = token_indices.cpu()

        norms = torch.norm(keys_cpu, p=2, dim=-1)
        min_norm = norms.min().item()
        max_norm = norms.max().item()

        def build_hnsw(
            bucket_keys: torch.Tensor, bucket_indices: torch.Tensor
        ) -> BucketEntry:
            n = bucket_keys.size(0)
            index = hnswlib.Index(space="ip", dim=head_dim)
            index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
            index.set_ef(ef_search)

            keys_np = bucket_keys.float().numpy()
            keys_np /= np.linalg.norm(keys_np, axis=1, keepdims=True).clip(min=1e-12)

            # Pass token positions as ids so knn_query returns token positions directly
            index.add_items(
                keys_np,
                bucket_indices.numpy().astype(np.int64),
            )
            return BucketEntry(hnsw=index, num_tokens=n)

        edges = torch.linspace(min_norm, max_norm, num_buckets + 1)
        bucket_ids = torch.bucketize(norms, edges[1:-1])
        for i in range(num_buckets):
            idx = torch.nonzero(bucket_ids == i, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                self.buckets.append(build_hnsw(keys_cpu[idx], token_indices_cpu[idx]))
                self.norm_ranges.append((edges[i].item(), edges[i + 1].item()))

    def search(self, query: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Args:
            query: [head_dim]
        Returns:
            token_indices: [<=top_k]
        """
        query_np = query.cpu().float().unsqueeze(0).numpy()
        query_np /= np.linalg.norm(query_np, axis=1, keepdims=True).clip(min=1e-12)
        all_labels = []
        all_scores = []

        # Allocate k proportionally by bucket norm midpoint: higher-norm buckets get
        mid_norms = [(lo + hi) / 2.0 for lo, hi in self.norm_ranges]
        total_weight = sum(mid_norms)

        for bucket, mid_norm in zip(self.buckets, mid_norms):
            k = max(1, round(top_k * mid_norm / total_weight))
            k = min(k, bucket.num_tokens)
            labels, distances = bucket.hnsw.knn_query(query_np, k=k)

            # Approximate attention score: mid_norm * cos_sim
            # space="ip" returns inner product (higher = more similar)
            scores = [mid_norm * d for d in distances[0].tolist()]
            all_labels.extend(labels[0].tolist())
            all_scores.extend(scores)

        # Sort by approximate attention score descending
        sorted_pairs = sorted(
            zip(all_scores, all_labels), key=lambda x: x[0], reverse=True
        )
        top_labels = [label for _, label in sorted_pairs[:top_k]]

        return torch.tensor(top_labels, dtype=torch.long)

    def get_num_buckets(self) -> int:
        return len(self.buckets)

    def get_bucket_num_tokens(self) -> List[int]:
        return [b.num_tokens for b in self.buckets]


class HierarchicalIndex:
    """
    Builds a two-level hierarchical index for each KV head independently.
    - Level 1: partition keys into buckets by L2 norm
    - Level 2: build an HNSW cosine similarity index within each bucket
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        num_norm_buckets: int = 10,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        device: str = "cpu",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_norm_buckets = num_norm_buckets
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.device = device

        # {batch_idx: {layer_idx: {head_idx: LayerHeadIndex}}}
        self.indices: Dict[int, Dict[int, Dict[int, LayerHeadIndex]]] = {}

        logger.info(
            f"HierarchicalIndex initialized: {num_layers} layers, "
            f"{num_kv_heads} heads, {num_norm_buckets} norm buckets, "
            f"M={M}, ef_construction={ef_construction}, ef_search={ef_search}"
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
        Args:
            keys: [num_tokens, head_dim]
            token_indices: [num_tokens]
        """
        index = LayerHeadIndex(
            keys=keys,
            token_indices=token_indices,
            num_buckets=self.num_norm_buckets,
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
        )
        self.indices.setdefault(batch_idx, {}).setdefault(layer_idx, {})[head_idx] = (
            index
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
        Args:
            query: [head_dim]
        Returns:
            token_indices: [<=top_k]
        """
        return self.indices[batch_idx][layer_idx][head_idx].search(query, top_k)

    def has_index(self, layer_idx: int, batch_idx: int = 0) -> bool:
        return batch_idx in self.indices and layer_idx in self.indices[batch_idx]

    def clear(self, batch_idx: Optional[int] = None):
        if batch_idx is not None:
            self.indices.pop(batch_idx, None)
        else:
            self.indices.clear()
