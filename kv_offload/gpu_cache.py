import logging
from typing import Optional

import torch
from flashinfer import single_decode_with_kv_cache, single_prefill_with_kv_cache

logger = logging.getLogger(__name__)


class GPUKVCache:
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

        # 为每一层创建cache [batch, 2, num_layers, heads, seq, dim]
        self.cache = torch.zeros(
            max_batch_size,
            2,  # 0=key, 1=value
            num_layers,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )

        # 记录每层当前的sequence length
        self.seq_lens = torch.zeros(
            max_batch_size, num_layers, dtype=torch.int32, device=device
        )

        # 标记哪些token位置是有效的（未被offload）
        self.valid_mask = torch.ones(
            max_batch_size,
            num_layers,
            max_seq_len,
            dtype=torch.bool,
            device=device,
        )

        logger.info(
            f"GPUKVCache initialized: {num_layers} layers, "
            f"{num_kv_heads} KV heads, head_dim={head_dim}"
        )

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        更新GPU cache
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
            batch_idx, 0, layer_idx, :, current_len : current_len + seq_len, :
        ] = key_states[0]
        self.cache[
            batch_idx, 1, layer_idx, :, current_len : current_len + seq_len, :
        ] = value_states[0]

        # 标记这些位置为有效
        self.valid_mask[batch_idx, layer_idx, current_len : current_len + seq_len] = (
            True
        )

        # 更新sequence length
        new_len = current_len + seq_len
        self.seq_lens[batch_idx][layer_idx] = new_len

        # 返回完整的KV，保留batch维度
        full_key = self.cache[batch_idx : batch_idx + 1, 0, layer_idx, :, :new_len, :]
        full_value = self.cache[batch_idx : batch_idx + 1, 1, layer_idx, :, :new_len, :]

        return full_key, full_value

    def get(
        self, layer_idx: int, batch_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取GPU上的KV
        Returns:
            key, value: [1, num_kv_heads, seq_len, head_dim]
        """
        seq_len = self.seq_lens[batch_idx][layer_idx].item()
        key = self.cache[batch_idx : batch_idx + 1, 0, layer_idx, :, :seq_len, :]
        value = self.cache[batch_idx : batch_idx + 1, 1, layer_idx, :, :seq_len, :]
        return key, value

    def get_valid_kv(
        self, layer_idx: int, batch_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取GPU上有效的KV（未被offload的）
        Returns:
            key: [1, num_kv_heads, valid_seq_len, head_dim]
            value: [1, num_kv_heads, valid_seq_len, head_dim]
            valid_indices: [valid_seq_len] token位置索引
        """
        seq_len = self.seq_lens[batch_idx][layer_idx].item()
        valid_mask = self.valid_mask[batch_idx, layer_idx, :seq_len]
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        # 提取有效的KV
        key = self.cache[
            batch_idx, 0, layer_idx, :, valid_indices, :
        ]  # [heads, valid_len, dim]
        value = self.cache[batch_idx, 1, layer_idx, :, valid_indices, :]

        # 添加batch维度
        key = key.unsqueeze(0)  # [1, heads, valid_len, dim]
        value = value.unsqueeze(0)

        return key, value, valid_indices

    def mark_offloaded(
        self, layer_idx: int, token_indices: torch.Tensor, batch_idx: int = 0
    ):
        """
        标记某些tokens已被offload到CPU
        Args:
            token_indices: [num_offload] 要offload的token位置
        """
        token_indices = token_indices.to(self.device)
        self.valid_mask[batch_idx, layer_idx, token_indices] = False
        logger.debug(
            f"Layer {layer_idx}: Marked {len(token_indices)} tokens as offloaded"
        )

    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        is_prefill: bool = False,
        batch_idx: int = 0,
        return_lse: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            layer_idx: 层索引
            query: [batch_size, num_heads, seq_len_q, head_dim]
            is_prefill: 是否为prefill阶段
            batch_idx: batch索引
            return_lse: 是否返回LSE用于后续合并
        Returns:
            output: prefill时 [batch, seq_len_q, num_heads, head_dim],
                    decode时  [batch, num_heads, 1, head_dim]
            lse: Optional[torch.Tensor]
        """
        key, value, _ = self.get_valid_kv(layer_idx, batch_idx)
        # key/value: [1, num_kv_heads, seq_len_k, head_dim]

        # FlashInfer 格式转换
        q = query.squeeze(0).transpose(0, 1)  # [seq_len_q, num_heads, head_dim]
        k = key.squeeze(0)  # [num_kv_heads, seq_len_k, head_dim]
        v = value.squeeze(0)  # [num_kv_heads, seq_len_k, head_dim]

        lse = None
        if is_prefill:
            if return_lse:
                output, lse = single_prefill_with_kv_cache(
                    q=q, k=k, v=v, kv_layout="HND", causal=True, return_lse=True
                )
            else:
                output = single_prefill_with_kv_cache(
                    q=q, k=k, v=v, kv_layout="HND", causal=True
                )
            # output: [seq_len_q, num_heads, head_dim]
            output = output.unsqueeze(0)
            # output: [1, seq_len_q, num_heads, head_dim]
        else:
            # Decode: q 需要是 [num_heads, head_dim]（无seq_len维度）
            q = q.squeeze(0)
            k = k.contiguous()
            v = v.contiguous()
            if return_lse:
                output, lse = single_decode_with_kv_cache(
                    q=q, k=k, v=v, kv_layout="HND", return_lse=True
                )
            else:
                output = single_decode_with_kv_cache(q=q, k=k, v=v, kv_layout="HND")
            # output: [num_heads, head_dim]
            output = output.unsqueeze(0).unsqueeze(2)
            # output: [1, num_heads, 1, head_dim]
            if lse is not None:
                lse = lse.unsqueeze(0).unsqueeze(-1)
                # lse: [num_heads] → [1, num_heads, 1]

        return output, lse

    def clear(self, batch_idx: Optional[int] = None):
        if batch_idx is not None:
            self.seq_lens[batch_idx] = 0
            self.cache[batch_idx, :, :, :, :, :] = 0
            self.valid_mask[batch_idx, :, :] = True
        else:
            self.seq_lens[:] = 0
            self.cache[:] = 0
            self.valid_mask[:] = True

    def get_seq_len(self, batch_idx: int = 0, layer_idx: int = 0) -> int:
        return int(self.seq_lens[batch_idx][layer_idx].item())

    def get_num_valid_tokens(self, layer_idx: int, batch_idx: int = 0) -> int:
        seq_len = self.get_seq_len(batch_idx, layer_idx)
        return int(self.valid_mask[batch_idx, layer_idx, :seq_len].sum().item())
