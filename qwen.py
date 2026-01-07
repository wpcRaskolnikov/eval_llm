import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from flashinfer import single_decode_with_kv_cache, single_prefill_with_kv_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Qwen3InferenceConfig:
    model_path: str = "/home/wpc/huggingface/Qwen3-8B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    max_batch_size: int = 1
    max_seq_len: int = 1024

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


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
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            key_states: [batch_size, num_kv_heads, seq_len, head_dim]
            value_states: [batch_size, num_kv_heads, seq_len, head_dim]
        Returns:
            完整的key和value: [batch_size, num_kv_heads, total_seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        current_len = self.seq_lens[batch_idx].item()

        # 写入新的KV到cache
        self.cache[
            layer_idx, 0, batch_idx, :, current_len : current_len + seq_len, :
        ] = key_states[0]
        self.cache[
            layer_idx, 1, batch_idx, :, current_len : current_len + seq_len, :
        ] = value_states[0]

        # 更新sequence length
        new_len = current_len + seq_len
        self.seq_lens[batch_idx] = new_len

        # 返回完整的KV (从0到new_len)
        full_key = self.cache[layer_idx, 0, batch_idx : batch_idx + 1, :, :new_len, :]
        full_value = self.cache[layer_idx, 1, batch_idx : batch_idx + 1, :, :new_len, :]

        return full_key, full_value

    def get(
        self, layer_idx: int, batch_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定层的完整KV cache

        Returns:
            key, value: [1, num_kv_heads, seq_len, head_dim]
        """
        seq_len = self.seq_lens[batch_idx].item()
        key = self.cache[layer_idx, 0, batch_idx : batch_idx + 1, :, :seq_len, :]
        value = self.cache[layer_idx, 1, batch_idx : batch_idx + 1, :, :seq_len, :]
        return key, value

    def clear(self, batch_idx: Optional[int] = None):
        """清空cache"""
        if batch_idx is not None:
            self.seq_lens[batch_idx] = 0
            self.cache[:, :, batch_idx, :, :, :] = 0
        else:
            self.seq_lens[:] = 0
            self.cache[:] = 0

    def get_seq_len(self, batch_idx: int = 0) -> int:
        """获取当前序列长度"""
        return self.seq_lens[batch_idx].item()


class Qwen3Inference:
    def __init__(self, config: Qwen3InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)

        logger.info(f"Loading model from {config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            dtype=config.dtype,
            device_map=config.device,
            trust_remote_code=True,
        )
        self.model.eval()

        model_config: Qwen3Config = self.model.config
        self.num_layers = model_config.num_hidden_layers
        self.num_attention_heads = model_config.num_attention_heads
        self.num_kv_heads = getattr(
            model_config, "num_key_value_heads", self.num_attention_heads
        )
        self.head_dim = model_config.head_dim
        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size

        # TODO
        self.kv_cache = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            dtype=config.dtype,
            device=self.device,
        )

        logger.info(
            f"Model loaded: {self.num_layers} layers, "
            f"{self.num_attention_heads} attention heads, "
            f"{self.num_kv_heads} KV heads, "
            f"head_dim={self.head_dim}, "
            f"hidden_size={self.hidden_size}"
        )

    def apply_rotary_emb(self, q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """
        Args:
            q: [batch_size, num_heads, seq_len, head_dim]
            k: [batch_size, num_kv_heads, seq_len, head_dim]
            cos: [batch_size, seq_len, head_dim]
            sin: [batch_size, seq_len, head_dim]
        """

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # cos, sin shape: [1, seq_len, head_dim]
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, num_heads, seq_len_q, head_dim]
            key: [batch_size, num_kv_heads, seq_len_k, head_dim]
            value: [batch_size, num_kv_heads, seq_len_k, head_dim]
        Returns:
            output: [batch_size, num_heads, seq_len_q, head_dim]
        """
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        _, num_kv_heads, seq_len_k, _ = key.shape

        # FlashInfer 格式转换
        q = query.squeeze(0)  # [num_heads, seq_len_q, head_dim]
        k = key.squeeze(0)  # [num_kv_heads, seq_len_k, head_dim]
        v = value.squeeze(0)  # [num_kv_heads, seq_len_k, head_dim]

        if is_prefill:
            output = single_prefill_with_kv_cache(
                q=q,
                k=k,
                v=v,
                kv_layout="HND",
            )
            # output: [seq_len_q, num_heads, head_dim]
            # 转换回: [batch, num_heads, seq_len_q, head_dim]
            output = output.unsqueeze(0)
        else:
            # # Decode: q需要是 [num_qo_heads, head_dim] (无seq_len维度！)
            q = q.squeeze(1).contiguous()  # [num_heads, head_dim]

            output = single_decode_with_kv_cache(
                q=q,
                k=k,
                v=v,
                kv_layout="HND",
            )
            # output: [num_heads, head_dim]
            # 转换回: [batch, num_heads, 1, head_dim]
            output = output.unsqueeze(0).unsqueeze(2)

        return output

    def _forward_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        layer = self.model.model.layers[layer_idx]

        residual = hidden_states

        # Pre-attention LayerNorm
        hidden_states = layer.input_layernorm(hidden_states)

        # Self-Attention
        attn = layer.self_attn

        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = attn.q_proj(hidden_states).view(hidden_shape)
        query_states = attn.q_norm(query_states).transpose(1, 2)
        # query_states: [batch, num_heads, seq_len, head_dim]

        key_states = attn.k_proj(hidden_states).view(hidden_shape)
        key_states = attn.k_norm(key_states).transpose(1, 2)
        # key_states: [batch, num_kv_heads, seq_len, head_dim]

        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # value_states: [batch, num_kv_heads, seq_len, head_dim]

        kv_seq_len = self.kv_cache.get_seq_len() + seq_len
        position_ids = torch.arange(
            kv_seq_len - seq_len,
            kv_seq_len,
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(value_states, position_ids)
        query_states, key_states = self.apply_rotary_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # 更新KV cache并获取完整的KV
        key_cache, value_cache = self.kv_cache.update(
            layer_idx, key_states, value_states
        )

        # 计算attention
        attn_output = self._compute_attention(
            query_states,
            key_cache,
            value_cache,
            layer_idx,
            is_prefill=is_prefill,
        )

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            batch_size, seq_len, self.num_attention_heads * self.head_dim
        )

        # Output projection
        attn_output = attn.o_proj(attn_output)
        # attn_output = F.linear(attn_output, attn.o_proj.weight)

        # Residual connection
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        logger.info(f"Prefill: processing {seq_len} tokens")

        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(input_ids)

            for layer_idx in range(self.num_layers):
                hidden_states = self._forward_layer(
                    hidden_states, layer_idx, is_prefill=True
                )

            hidden_states = self.model.model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)

        return logits

    def decode_step(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        Decode阶段: 生成下一个token

        Args:
            token_id: [batch_size, 1] - 当前token

        Returns:
            logits: [batch_size, 1, vocab_size]
        """
        with torch.no_grad():
            # Embedding
            hidden_states = self.model.model.embed_tokens(token_id)

            # 逐层前向传播
            for layer_idx in range(self.num_layers):
                hidden_states = self._forward_layer(
                    hidden_states, layer_idx, is_prefill=False
                )

            # Final LayerNorm
            hidden_states = self.model.model.norm(hidden_states)

            # LM head
            logits = self.model.lm_head(hidden_states)

        return logits

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, vocab_size]
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数

        Returns:
            token: [batch_size]
        """
        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        stream: bool = False,
    ) -> str:
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k

        self.kv_cache.clear()

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)

        logger.info(f"Input: {prompt}")
        logger.info(f"Input tokens: {input_ids.shape[1]}")

        # Prefill
        prefill_start = time.time()
        logits = self.prefill(input_ids)
        prefill_time = time.time() - prefill_start
        logger.info(f"Prefill completed in {prefill_time:.3f}s")

        # 获取第一个生成的token
        next_token_logits = logits[:, -1, :]
        next_token = self._sample_token(next_token_logits, temperature, top_p, top_k)
        generated_tokens = [next_token.item()]

        if stream:
            print(
                self.tokenizer.decode([next_token.item()], skip_special_tokens=True),
                end="",
                flush=True,
            )

        # Decode loop
        decode_start = time.time()
        for i in range(max_new_tokens - 1):
            # Decode one token
            token_input = next_token.unsqueeze(-1)
            logits = self.decode_step(token_input)

            # Sample next token
            next_token_logits = logits[:, -1, :]
            next_token = self._sample_token(
                next_token_logits, temperature, top_p, top_k
            )
            generated_tokens.append(next_token.item())

            if stream:
                print(
                    self.tokenizer.decode(
                        [next_token.item()], skip_special_tokens=True
                    ),
                    end="",
                    flush=True,
                )

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                logger.info(f"EOS reached at step {i + 1}")
                break

            # Progress logging
            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i + 1} tokens")

        decode_time = time.time() - decode_start
        tokens_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0

        if stream:
            print()  # New line after streaming

        logger.info(
            f"Decode: {len(generated_tokens)} tokens in {decode_time:.3f}s "
            f"({tokens_per_sec:.2f} tokens/s)"
        )

        # Decode generated tokens
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return prompt + generated_text


def main():
    """示例用法"""
    # 配置
    config = Qwen3InferenceConfig(
        model_path="/home/wpc/huggingface/Qwen3-8B",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

    # 初始化推理引擎
    logger.info("Initializing Qwen3 Inference Engine")
    engine = Qwen3Inference(config)

    # 测试prompt
    prompt = "人工智能的未来发展方向是"

    logger.info("\n" + "=" * 70)
    logger.info("Starting Generation")
    logger.info("=" * 70)

    # 生成
    result = engine.generate(
        prompt=prompt,
        max_new_tokens=100,
        stream=True,
    )

    print("\n" + "=" * 70)
    print("Complete Result:")
    print("=" * 70)
    print(result)


if __name__ == "__main__":
    main()
