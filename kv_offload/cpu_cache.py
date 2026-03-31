import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class LayerCache:
    key_cache: torch.Tensor  # [num_kv_heads, num_tokens, head_dim]
    value_cache: torch.Tensor  # [num_kv_heads, num_tokens, head_dim]
    token_indices: torch.Tensor  # [num_tokens]


class CPUKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = True,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.pin_memory = pin_memory

        # {batch_idx: {layer_idx: LayerCache}}
        self.storage: Dict[int, Dict[int, LayerCache]] = {}

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
        keys_cpu = keys.cpu()
        values_cpu = values.cpu()
        token_indices_cpu = token_indices.cpu()

        if self.pin_memory:
            keys_cpu = keys_cpu.pin_memory()
            values_cpu = values_cpu.pin_memory()
            token_indices_cpu = token_indices_cpu.pin_memory()

        if keys_cpu.dtype != self.dtype:
            keys_cpu = keys_cpu.to(self.dtype)
            values_cpu = values_cpu.to(self.dtype)

        self.storage.setdefault(batch_idx, {})[layer_idx] = LayerCache(
            key_cache=keys_cpu,
            value_cache=values_cpu,
            token_indices=token_indices_cpu,
        )

    def get(
        self, layer_idx: int, batch_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.storage[batch_idx][layer_idx]
        return data.key_cache, data.value_cache, data.token_indices

    def get_by_indices(
        self,
        layer_idx: int,
        retrieve_indices: torch.Tensor,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.storage[batch_idx][layer_idx]
        positions = torch.searchsorted(data.token_indices, retrieve_indices)
        return data.key_cache[:, positions, :], data.value_cache[:, positions, :]

    def has_data(self, batch_idx: int = 0) -> bool:
        return batch_idx in self.storage

    def get_num_tokens(self, batch_idx: int = 0) -> int:
        if batch_idx not in self.storage:
            return 0
        return next(iter(self.storage[batch_idx].values())).key_cache.size(1)

    def clear(self, batch_idx: Optional[int] = None):
        if batch_idx is not None:
            self.storage.pop(batch_idx, None)
        else:
            self.storage.clear()

    def get_memory_usage_mb(self) -> float:
        total_bytes = 0
        for layers in self.storage.values():
            for data in layers.values():
                total_bytes += data.key_cache.element_size() * data.key_cache.numel()
                total_bytes += (
                    data.value_cache.element_size() * data.value_cache.numel()
                )
                total_bytes += (
                    data.token_indices.element_size() * data.token_indices.numel()
                )
        return total_bytes / (1024 * 1024)
