import logging
from typing import Literal, Optional, Tuple, overload

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

        # [batch, num_layers, num_kv_heads, max_seq_len, head_dim]
        self.key_cache = torch.zeros(
            max_batch_size,
            num_layers,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.value_cache = torch.zeros(
            max_batch_size,
            num_layers,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )

        self.seq_lens = torch.zeros(
            max_batch_size, num_layers, dtype=torch.int32, device=device
        )
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
        seq_len = key_states.shape[2]
        current_len = int(self.seq_lens[batch_idx, layer_idx])

        self.key_cache[
            batch_idx, layer_idx, :, current_len : current_len + seq_len, :
        ] = key_states[0]
        self.value_cache[
            batch_idx, layer_idx, :, current_len : current_len + seq_len, :
        ] = value_states[0]
        self.valid_mask[batch_idx, layer_idx, current_len : current_len + seq_len] = (
            True
        )
        new_len = current_len + seq_len
        self.seq_lens[batch_idx][layer_idx] = new_len

        # return full KV with batch dim preserved
        full_key = self.key_cache[batch_idx : batch_idx + 1, layer_idx, :, :new_len, :]
        full_value = self.value_cache[
            batch_idx : batch_idx + 1, layer_idx, :, :new_len, :
        ]

        return full_key, full_value

    def get(
        self, layer_idx: int, batch_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(self.seq_lens[batch_idx, layer_idx])
        key = self.key_cache[
            batch_idx : batch_idx + 1, layer_idx, :, :seq_len, :
        ]  # [1, num_kv_heads, seq_len, head_dim]
        value = self.value_cache[batch_idx : batch_idx + 1, layer_idx, :, :seq_len, :]
        return key, value

    def get_valid_kv(
        self, layer_idx: int, batch_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = int(self.seq_lens[batch_idx, layer_idx])
        valid_mask = self.valid_mask[batch_idx, layer_idx, :seq_len]
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        key = self.key_cache[
            batch_idx, layer_idx, :, valid_indices, :
        ]  # [heads, valid_len, dim]
        value = self.value_cache[batch_idx, layer_idx, :, valid_indices, :]

        key = key.unsqueeze(0)  # [1, heads, valid_len, dim]
        value = value.unsqueeze(0)

        return key, value, valid_indices

    def mark_offloaded(
        self, layer_idx: int, token_indices: torch.Tensor, batch_idx: int = 0
    ):
        token_indices = token_indices.to(self.device)
        self.valid_mask[batch_idx, layer_idx, token_indices] = False

    @overload
    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        is_prefill: bool,
        return_lse: Literal[False] = ...,
        batch_idx: int = ...,
    ) -> Tuple[torch.Tensor, None]: ...

    @overload
    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        is_prefill: bool,
        return_lse: Literal[True],
        batch_idx: int = ...,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        is_prefill: bool,
        return_lse: bool = False,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key, value, _ = self.get_valid_kv(layer_idx, batch_idx)

        # convert to FlashInfer layout
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
            output = output.unsqueeze(0)  # [1, seq_len_q, num_heads, head_dim]
        else:
            q = q.squeeze(0)  # [num_heads, head_dim]
            k = k.contiguous()
            v = v.contiguous()
            if return_lse:
                output, lse = single_decode_with_kv_cache(
                    q=q, k=k, v=v, kv_layout="HND", return_lse=True
                )
            else:
                output = single_decode_with_kv_cache(q=q, k=k, v=v, kv_layout="HND")
            output = output.unsqueeze(0).unsqueeze(1)
            # output: [num_heads, head_dim] -> [1, 1, num_heads, head_dim]
            if lse is not None:
                lse = lse.unsqueeze(0).unsqueeze(-1)
                # lse: [num_heads] -> [1, num_heads, 1]

        return output, lse

    def clear(self, batch_idx: Optional[int] = None):
        if batch_idx is not None:
            self.seq_lens[batch_idx] = 0
            self.key_cache[batch_idx] = 0
            self.value_cache[batch_idx] = 0
            self.valid_mask[batch_idx, :, :] = True
        else:
            self.seq_lens[:] = 0
            self.key_cache[:] = 0
            self.value_cache[:] = 0
            self.valid_mask[:] = True

    def get_seq_len(self, layer_idx: int, batch_idx: int = 0) -> int:
        return int(self.seq_lens[batch_idx, layer_idx])

    def get_num_valid_tokens(self, layer_idx: int, batch_idx: int = 0) -> int:
        seq_len = self.get_seq_len(layer_idx, batch_idx)
        return int(self.valid_mask[batch_idx, layer_idx, :seq_len].sum())
