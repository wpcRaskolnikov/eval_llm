import logging
from typing import Optional

import torch
import torch.nn.functional as F

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

        # 为每一层创建cache [num_layers, 2, batch, heads, seq, dim]
        self.cache = torch.zeros(
            num_layers,
            2,  # 0=key, 1=value
            max_batch_size,
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
            num_layers,
            max_batch_size,
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
            layer_idx, 0, batch_idx, :, current_len : current_len + seq_len, :
        ] = key_states[0]
        self.cache[
            layer_idx, 1, batch_idx, :, current_len : current_len + seq_len, :
        ] = value_states[0]

        # 标记这些位置为有效
        self.valid_mask[layer_idx, batch_idx, current_len : current_len + seq_len] = (
            True
        )

        # 更新sequence length
        new_len = current_len + seq_len
        self.seq_lens[batch_idx][layer_idx] = new_len

        # 返回完整的KV
        full_key = self.cache[layer_idx, 0, batch_idx : batch_idx + 1, :, :new_len, :]
        full_value = self.cache[layer_idx, 1, batch_idx : batch_idx + 1, :, :new_len, :]

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
        key = self.cache[layer_idx, 0, batch_idx : batch_idx + 1, :, :seq_len, :]
        value = self.cache[layer_idx, 1, batch_idx : batch_idx + 1, :, :seq_len, :]
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
        valid_mask = self.valid_mask[layer_idx, batch_idx, :seq_len]
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        # 提取有效的KV
        key = self.cache[
            layer_idx, 0, batch_idx, :, valid_indices, :
        ]  # [heads, valid_len, dim]
        value = self.cache[layer_idx, 1, batch_idx, :, valid_indices, :]

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
        self.valid_mask[layer_idx, batch_idx, token_indices] = False
        logger.debug(
            f"Layer {layer_idx}: Marked {len(token_indices)} tokens as offloaded"
        )

    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        num_q_heads: int,
        batch_idx: int = 0,
        return_lse: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算GPU上有效KV的局部attention
        Args:
            query: [batch_size, num_q_heads, 1, head_dim] decode时的query
            num_q_heads: query heads数量（可能和kv_heads不同，GQA）
            return_lse: 是否返回LSE用于后续合并
        Returns:
            output: [batch_size, num_q_heads, 1, head_dim]
            lse: [batch_size, num_q_heads, 1] 如果return_lse=True
        """
        # 获取GPU上有效的KV
        key, value, valid_indices = self.get_valid_kv(layer_idx, batch_idx)
        # key/value: [1, num_kv_heads, valid_len, head_dim]

        if key.size(2) == 0:  # 没有有效的KV
            batch_size = query.size(0)
            output = torch.zeros_like(query)
            if return_lse:
                lse = torch.full(
                    (batch_size, num_q_heads, 1),
                    float("-inf"),
                    dtype=torch.float32,
                    device=query.device,
                )
                return output, lse
            return output, None

        # GQA: 扩展KV heads到匹配query heads
        num_kv_heads = key.size(1)
        if num_q_heads != num_kv_heads:
            # Repeat KV heads
            n_rep = num_q_heads // num_kv_heads
            key = key.repeat(1, n_rep, 1, 1)  # [1, num_q_heads, valid_len, head_dim]
            value = value.repeat(1, n_rep, 1, 1)

        # 计算attention: Q @ K^T / sqrt(d)
        # query: [1, num_q_heads, 1, head_dim]
        # key: [1, num_q_heads, valid_len, head_dim]
        scale = 1.0 / (self.head_dim**0.5)

        # [1, num_q_heads, 1, valid_len]
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        # 计算LSE (log-sum-exp) 用于后续合并
        if return_lse:
            # LSE = max + log(sum(exp(x - max)))
            max_score = torch.max(attn_weights, dim=-1, keepdim=True)[
                0
            ]  # [1, heads, 1, 1]
            lse = max_score.squeeze(-1) + torch.log(
                torch.sum(torch.exp(attn_weights - max_score), dim=-1, keepdim=True)
            ).squeeze(-1)  # [1, heads, 1]

        # Softmax + 乘以value
        attn_weights = F.softmax(attn_weights, dim=-1)  # [1, num_q_heads, 1, valid_len]
        output = torch.matmul(attn_weights, value)  # [1, num_q_heads, 1, head_dim]

        if return_lse:
            return output, lse
        return output, None

    def clear(self, batch_idx: Optional[int] = None):
        if batch_idx is not None:
            self.seq_lens[batch_idx] = 0
            self.cache[:, :, batch_idx, :, :, :] = 0
            self.valid_mask[:, batch_idx, :] = True
        else:
            self.seq_lens[:] = 0
            self.cache[:] = 0
            self.valid_mask[:] = True

    def get_seq_len(self, batch_idx: int = 0, layer_idx: int = 0) -> int:
        return int(self.seq_lens[batch_idx][layer_idx].item())

    def get_num_valid_tokens(self, layer_idx: int, batch_idx: int = 0) -> int:
        """获取GPU上有效token数量"""
        seq_len = self.get_seq_len(batch_idx, layer_idx)
        return int(self.valid_mask[layer_idx, batch_idx, :seq_len].sum().item())
