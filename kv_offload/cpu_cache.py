import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CPUKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        pin_memory: bool = True,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.pin_memory = pin_memory

        # 存储结构: {(layer_idx, batch_idx): {"keys": tensor, "values": tensor, "token_indices": tensor}}
        # keys/values: [num_kv_heads, num_offloaded_tokens, head_dim]
        # token_indices: [num_offloaded_tokens] 原始位置索引
        self.storage: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        logger.info(
            f"CPUKVCache initialized: {num_layers} layers, "
            f"{num_kv_heads} KV heads, pin_memory={pin_memory}"
        )

    def store(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_indices: torch.Tensor,
        batch_idx: int = 0,
    ):
        """
        存储offloaded的KV到CPU
        Args:
            keys: [num_kv_heads, num_tokens, head_dim] 在GPU上
            values: [num_kv_heads, num_tokens, head_dim] 在GPU上
            token_indices: [num_tokens] 这些KV对应的原始token位置
            batch_idx: batch索引
        """
        # 转到CPU并使用pin_memory加速后续传输
        keys_cpu = keys.cpu()
        values_cpu = values.cpu()
        token_indices_cpu = token_indices.cpu()

        if self.pin_memory:
            keys_cpu = keys_cpu.pin_memory()
            values_cpu = values_cpu.pin_memory()
            token_indices_cpu = token_indices_cpu.pin_memory()

        # 转换为指定dtype以节省内存
        if keys_cpu.dtype != self.dtype:
            keys_cpu = keys_cpu.to(self.dtype)
            values_cpu = values_cpu.to(self.dtype)

        key = (layer_idx, batch_idx)
        self.storage[key] = {
            "keys": keys_cpu,
            "values": values_cpu,
            "token_indices": token_indices_cpu,
        }

        logger.debug(
            f"Stored {keys.size(1)} tokens to CPU cache for layer {layer_idx}, "
            f"batch {batch_idx}"
        )

    def get(
        self, layer_idx: int, batch_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取该层所有offloaded的KV
        Returns:
            keys: [num_kv_heads, num_tokens, head_dim] 在CPU上
            values: [num_kv_heads, num_tokens, head_dim] 在CPU上
            token_indices: [num_tokens] token位置
        """
        key = (layer_idx, batch_idx)
        if key not in self.storage:
            # 返回空tensor
            empty = torch.empty(self.num_kv_heads, 0, self.head_dim, dtype=self.dtype)
            empty_indices = torch.empty(0, dtype=torch.long)
            return empty, empty, empty_indices

        data = self.storage[key]
        return data["keys"], data["values"], data["token_indices"]

    def get_subset(
        self,
        layer_idx: int,
        retrieve_indices: torch.Tensor,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定token indices的KV
        Args:
            retrieve_indices: [num_retrieve] 要检索的token位置（原始位置）
        Returns:
            keys: [num_kv_heads, num_retrieve, head_dim]
            values: [num_kv_heads, num_retrieve, head_dim]
        """
        key = (layer_idx, batch_idx)
        if key not in self.storage or len(retrieve_indices) == 0:
            empty = torch.empty(self.num_kv_heads, 0, self.head_dim, dtype=self.dtype)
            return empty, empty

        data = self.storage[key]
        stored_indices = data["token_indices"]  # [num_stored]

        # 找到retrieve_indices在stored_indices中的位置
        # 使用broadcasting找交集
        # retrieve_indices: [num_retrieve]
        # stored_indices: [num_stored]
        mask = retrieve_indices.unsqueeze(1) == stored_indices.unsqueeze(
            0
        )  # [num_retrieve, num_stored]
        positions = torch.nonzero(mask, as_tuple=False)[:, 1]  # [num_found]

        if len(positions) == 0:
            empty = torch.empty(self.num_kv_heads, 0, self.head_dim, dtype=self.dtype)
            return empty, empty

        # 提取对应的KV
        keys = data["keys"][:, positions, :]  # [heads, num_found, dim]
        values = data["values"][:, positions, :]

        return keys, values

    def get_keys_for_indexing(
        self, layer_idx: int, batch_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取keys用于建立索引
        Returns:
            keys: [num_kv_heads, num_tokens, head_dim]
            token_indices: [num_tokens]
        """
        keys, _, token_indices = self.get(layer_idx, batch_idx)
        return keys, token_indices

    def has_data(self, layer_idx: int, batch_idx: int = 0) -> bool:
        """检查是否有offloaded数据"""
        key = (layer_idx, batch_idx)
        return key in self.storage and self.storage[key]["keys"].size(1) > 0

    def get_num_tokens(self, layer_idx: int, batch_idx: int = 0) -> int:
        """获取offloaded token数量"""
        key = (layer_idx, batch_idx)
        if key not in self.storage:
            return 0
        return self.storage[key]["keys"].size(1)

    def clear(self, layer_idx: Optional[int] = None, batch_idx: Optional[int] = None):
        if layer_idx is not None and batch_idx is not None:
            key = (layer_idx, batch_idx)
            if key in self.storage:
                del self.storage[key]
        elif layer_idx is not None:
            # 清空该层所有batch
            keys_to_delete = [k for k in self.storage.keys() if k[0] == layer_idx]
            for k in keys_to_delete:
                del self.storage[k]
        elif batch_idx is not None:
            # 清空该batch所有层
            keys_to_delete = [k for k in self.storage.keys() if k[1] == batch_idx]
            for k in keys_to_delete:
                del self.storage[k]
        else:
            # 清空所有
            self.storage.clear()

    def get_memory_usage_mb(self) -> float:
        """获取CPU cache内存使用量（MB）"""
        total_bytes = 0
        for data in self.storage.values():
            total_bytes += data["keys"].element_size() * data["keys"].numel()
            total_bytes += data["values"].element_size() * data["values"].numel()
            total_bytes += (
                data["token_indices"].element_size() * data["token_indices"].numel()
            )
        return total_bytes / (1024 * 1024)
