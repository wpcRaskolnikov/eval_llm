import logging
import time
from dataclasses import dataclass
from typing import Optional, cast

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from kv_offload.gpu_cache import GPUKVCache

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Qwen3OriginInferenceConfig:
    model_path: str = "/home/wpc/huggingface/Qwen3-8B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    max_batch_size: int = 1
    max_seq_len: int = 4096

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


class Qwen3OriginInference:
    def __init__(self, config: Qwen3OriginInferenceConfig):
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

        model_config: Qwen3Config = cast(Qwen3Config, self.model.config)
        self.num_layers = model_config.num_hidden_layers
        self.num_attention_heads = model_config.num_attention_heads
        self.num_kv_heads = getattr(
            model_config, "num_key_value_heads", self.num_attention_heads
        )
        self.head_dim = model_config.head_dim
        self.hidden_size = model_config.hidden_size

        self.kv_cache = GPUKVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            dtype=config.dtype,
            device=self.device,
        )

        logger.info(
            f"Model loaded: {self.num_layers} layers, {self.num_attention_heads} attention heads, "
            f"{self.num_kv_heads} KV heads, head_dim={self.head_dim}, hidden_size={self.hidden_size}"
        )

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

        # ---- Self-Attention ----
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

        # RoPE
        current_len = self.kv_cache.get_seq_len(layer_idx=layer_idx)
        position_ids = torch.arange(
            current_len,
            current_len + seq_len,
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 写入 cache，再用 FlashInfer 计算 attention
        self.kv_cache.update(layer_idx, key_states, value_states)
        attn_output, _ = self.kv_cache.compute_attention(
            layer_idx, query_states, is_prefill=is_prefill
        )

        # ---- Reshape 回 [batch, seq_len, hidden_size] ----
        # prefill:  [1, seq_len, num_heads, head_dim]
        # decode:   [1, 1, num_heads, head_dim]
        attn_output = attn_output.view(
            batch_size, seq_len, self.num_attention_heads * self.head_dim
        )

        # Output projection + residual
        hidden_states = residual + attn.o_proj(attn_output)

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)

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
        Args:
            token_id: [batch_size, 1]
        Returns:
            logits: [batch_size, 1, vocab_size]
        """
        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(token_id)

            for layer_idx in range(self.num_layers):
                hidden_states = self._forward_layer(
                    hidden_states, layer_idx, is_prefill=False
                )

            hidden_states = self.model.model.norm(hidden_states)
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
        Returns:
            token: [batch_size]
        """
        if temperature == 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v[:, -1].unsqueeze(1)
            logits[logits < pivot] = float("-Inf")

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k

        self.kv_cache.clear()

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        input_ids = tokenized.input_ids.to(self.device)

        logger.info(f"Input: {prompt}")
        logger.info(f"Input tokens: {input_ids.shape[1]}")

        # ---- Prefill ----
        prefill_start = time.time()
        logits = self.prefill(input_ids)
        prefill_time = time.time() - prefill_start
        logger.info(f"Prefill completed in {prefill_time:.3f}s")

        # 第一个生成的 token
        next_token = self._sample_token(logits[:, -1, :], temperature, top_p, top_k)
        generated_tokens = [next_token.item()]

        stream_printed_len = 0
        if stream:
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(text, end="", flush=True)
            stream_printed_len = len(text)

        # ---- Decode loop ----
        decode_start = time.time()
        for i in range(max_new_tokens - 1):
            logits = self.decode_step(next_token.unsqueeze(-1))
            next_token = self._sample_token(logits[:, -1, :], temperature, top_p, top_k)
            generated_tokens.append(next_token.item())

            if stream:
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(text[stream_printed_len:], end="", flush=True)
                stream_printed_len = len(text)

            if next_token.item() == self.tokenizer.eos_token_id:
                logger.info(f"EOS reached at step {i + 1}")
                break

            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i + 1} tokens")

        decode_time = time.time() - decode_start
        tokens_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0

        if stream:
            print()

        logger.info(
            f"Decode: {len(generated_tokens)} tokens in {decode_time:.3f}s ({tokens_per_sec:.2f} tokens/s)"
        )

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    config = Qwen3OriginInferenceConfig(
        model_path="/home/wpc/huggingface/Qwen3-8B",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

    logger.info("Initializing Qwen3 Origin Inference Engine (pure GPU)")
    engine = Qwen3OriginInference(config)

    prompt = "hello"
    with open("prompt.txt", "r") as f:
        prompt = f.read()

    logger.info("\n" + "=" * 70)
    logger.info("Starting Generation")
    logger.info("=" * 70)

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
