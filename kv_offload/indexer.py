import logging
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np
import torch

logger = logging.getLogger(__name__)


class HierarchicalIndex:
    """
    为每个attention head独立建立分层索引
    - 第一层：按key范数均匀分桶
    - 第二层：每个桶内建立HNSW余弦相似度索引
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

        self.indices: Dict[Tuple[int, int, int], "LayerHeadIndex"] = {}

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
        为指定layer和head建立索引
        Args:
            keys: [num_tokens, head_dim] 该head的所有keys（在CPU上）
            token_indices: [num_tokens] 对应的token位置
        """
        key = (layer_idx, head_idx, batch_idx)
        index = LayerHeadIndex(
            keys=keys,
            token_indices=token_indices,
            num_buckets=self.num_norm_buckets,
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
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
        Args:
            query: [head_dim]
        Returns:
            token_indices: [<=top_k]
        """
        key = (layer_idx, head_idx, batch_idx)
        if key not in self.indices:
            return torch.empty(0, dtype=torch.long)
        return self.indices[key].search(query, top_k)

    def has_index(self, layer_idx: int, head_idx: int, batch_idx: int = 0) -> bool:
        return (layer_idx, head_idx, batch_idx) in self.indices

    def clear(self, layer_idx: Optional[int] = None, batch_idx: Optional[int] = None):
        if layer_idx is not None and batch_idx is not None:
            keys_to_delete = [
                k for k in self.indices if k[0] == layer_idx and k[2] == batch_idx
            ]
            for k in keys_to_delete:
                del self.indices[k]
        elif layer_idx is not None:
            keys_to_delete = [k for k in self.indices if k[0] == layer_idx]
            for k in keys_to_delete:
                del self.indices[k]
        elif batch_idx is not None:
            keys_to_delete = [k for k in self.indices if k[2] == batch_idx]
            for k in keys_to_delete:
                del self.indices[k]
        else:
            self.indices.clear()


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
            num_buckets: 范数分桶数量
            M/ef_construction/ef_search: HNSW 参数
        """
        self.ef_search = ef_search
        self.buckets: List[dict] = []  # [{"hnsw": Index, "size": int}]
        self.norm_ranges: List[Tuple[float, float]] = []

        if keys.size(0) == 0:
            return

        head_dim = keys.shape[1]
        keys_cpu = keys.cpu()
        token_indices_cpu = token_indices.cpu()

        norms = torch.norm(keys_cpu, p=2, dim=-1)
        min_norm = norms.min().item()
        max_norm = norms.max().item()

        def build_hnsw(bkeys: torch.Tensor, bindices: torch.Tensor) -> dict:
            n = bkeys.size(0)
            index = hnswlib.Index(space="cosine", dim=head_dim)
            index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
            index.set_ef(ef_search)
            index.add_items(
                bkeys.float().numpy(),
                bindices.numpy().astype(np.int64),
            )
            return {"hnsw": index, "size": n}

        if max_norm - min_norm < 1e-6:
            self.buckets = [build_hnsw(keys_cpu, token_indices_cpu)]
            self.norm_ranges = [(min_norm, max_norm)]
        else:
            bucket_width = (max_norm - min_norm) / num_buckets
            for i in range(num_buckets):
                b_min = min_norm + i * bucket_width
                b_max = (
                    min_norm + (i + 1) * bucket_width
                    if i < num_buckets - 1
                    else max_norm + 1e-6
                )
                mask = (norms >= b_min) & (norms < b_max)
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    self.buckets.append(
                        build_hnsw(keys_cpu[idx], token_indices_cpu[idx])
                    )
                    self.norm_ranges.append((b_min, b_max))

    def search(self, query: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        每个桶用HNSW检索top-k，跨桶按距离合并后返回全局top-k
        Args:
            query: [head_dim]
        Returns:
            token_indices: [<=top_k]
        """
        if not self.buckets:
            return torch.empty(0, dtype=torch.long)

        query_np = query.cpu().float().numpy().reshape(1, -1)
        all_labels = []
        all_distances = []

        for bucket in self.buckets:
            k = min(top_k, bucket["size"])
            labels, distances = bucket["hnsw"].knn_query(query_np, k=k)
            all_labels.extend(labels[0].tolist())
            all_distances.extend(distances[0].tolist())

        # hnswlib cosine distance = 1 - cos_sim，越小越相似，升序取前 top_k
        sorted_pairs = sorted(zip(all_distances, all_labels), key=lambda x: x[0])
        top_labels = [label for _, label in sorted_pairs[:top_k]]

        return torch.tensor(top_labels, dtype=torch.long)

    def get_num_buckets(self) -> int:
        return len(self.buckets)

    def get_bucket_sizes(self) -> List[int]:
        return [b["size"] for b in self.buckets]
