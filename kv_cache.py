import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class KVCacheManager:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device

        # 为每一层创建cache
        self.cache = torch.zeros(
            num_layers,
            2,
            max_batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )

        # 记录每一层当前的sequence length
        self.seq_lens = torch.zeros(
            max_batch_size, num_layers, dtype=torch.int32, device=device
        )

        logger.info(
            f"KVCacheManager initialized: {num_layers} layers, "
            f"{num_kv_heads} KV heads, head_dim={head_dim}, "
            f"max_batch_size={max_batch_size}, max_seq_len={max_seq_len}"
        )

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            key_states: [batch_size, num_kv_heads, seq_len, head_dim]
            value_states: [batch_size, num_kv_heads, seq_len, head_dim]
        Returns:
            完整的key和value: [batch_size, num_kv_heads, total_seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        current_len = self.seq_lens[batch_idx][layer_idx].item()

        # 写入新的KV到cache
        self.cache[
            layer_idx, 0, batch_idx, :, current_len : current_len + seq_len, :
        ] = key_states[0]
        self.cache[
            layer_idx, 1, batch_idx, :, current_len : current_len + seq_len, :
        ] = value_states[0]

        # 更新sequence length
        new_len = current_len + seq_len
        self.seq_lens[batch_idx][layer_idx] = new_len

        # 返回完整的KV (从0到new_len)
        full_key = self.cache[layer_idx, 0, batch_idx : batch_idx + 1, :, :new_len, :]
        full_value = self.cache[layer_idx, 1, batch_idx : batch_idx + 1, :, :new_len, :]

        return full_key, full_value

    def get(
        self, layer_idx: int, batch_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            key, value: [1, num_kv_heads, seq_len, head_dim]
        """
        seq_len = self.seq_lens[batch_idx][layer_idx].item()
        key = self.cache[layer_idx, 0, batch_idx : batch_idx + 1, :, :seq_len, :]
        value = self.cache[layer_idx, 1, batch_idx : batch_idx + 1, :, :seq_len, :]
        return key, value

    def clear(self, batch_idx: Optional[int] = None):
        if batch_idx is not None:
            self.seq_lens[batch_idx] = 0
            self.cache[:, :, batch_idx, :, :, :] = 0
        else:
            self.seq_lens[:] = 0
            self.cache[:] = 0

    def get_seq_len(self, batch_idx: int = 0, layer_idx: int = 0) -> int:
        return int(self.seq_lens[batch_idx][layer_idx].item())
