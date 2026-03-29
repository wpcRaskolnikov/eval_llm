"""HNSW 索引：为每个 attention head 独立建立近似最近邻索引"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import hnswlib

logger = logging.getLogger(__name__)


class HierarchicalIndex:
    """
    为每个 attention head 独立建立 HNSW 近似最近邻索引
    使用余弦相似度空间，支持 O(log N) 的 top-k 检索
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        device: str = "cpu",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.device = device

        self.indices: Dict[Tuple[int, int, int], "LayerHeadIndex"] = {}

        logger.info(
            f"HierarchicalIndex (HNSW) initialized: {num_layers} layers, "
            f"{num_kv_heads} heads, head_dim={head_dim}, "
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
        为指定 layer 和 head 建立 HNSW 索引
        Args:
            keys: [num_tokens, head_dim]
            token_indices: [num_tokens] 对应的原始 token 位置
        """
        key = (layer_idx, head_idx, batch_idx)
        self.indices[key] = LayerHeadIndex(
            keys=keys,
            token_indices=token_indices,
            head_dim=self.head_dim,
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
        )
        logger.debug(
            f"Built HNSW index for layer {layer_idx}, head {head_idx}: "
            f"{len(token_indices)} tokens"
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
        检索与 query 最相似的 top-k 个 token 位置
        Args:
            query: [head_dim]
            top_k: 返回数量
        Returns:
            token_indices: [top_k]
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
        elif layer_idx is not None:
            keys_to_delete = [k for k in self.indices if k[0] == layer_idx]
        elif batch_idx is not None:
            keys_to_delete = [k for k in self.indices if k[2] == batch_idx]
        else:
            self.indices.clear()
            return
        for k in keys_to_delete:
            del self.indices[k]


class LayerHeadIndex:
    """单个 layer-head 的 HNSW 索引"""

    def __init__(
        self,
        keys: torch.Tensor,
        token_indices: torch.Tensor,
        head_dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        """
        Args:
            keys: [num_tokens, head_dim]  可在 GPU 或 CPU 上
            token_indices: [num_tokens]   原始 token 位置（作为 HNSW label）
        """
        self.num_tokens = keys.shape[0]

        if self.num_tokens == 0:
            self.index = None
            return

        self.index = hnswlib.Index(space="cosine", dim=head_dim)
        self.index.init_index(
            max_elements=self.num_tokens,
            ef_construction=ef_construction,
            M=M,
        )
        self.index.set_ef(ef_search)

        keys_np = keys.cpu().float().numpy()
        labels = token_indices.cpu().numpy().astype(np.int64)
        self.index.add_items(keys_np, labels)

    def search(self, query: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Args:
            query: [head_dim]
            top_k: 返回数量
        Returns:
            token_indices: [min(top_k, num_tokens)]
        """
        if self.index is None or top_k == 0:
            return torch.empty(0, dtype=torch.long)

        k = min(top_k, self.num_tokens)
        query_np = query.cpu().float().numpy().reshape(1, -1)
        labels, _ = self.index.knn_query(query_np, k=k)
        return torch.tensor(labels[0], dtype=torch.long)
