import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional, cast

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from kv_offload import HybridKVCacheManager

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

    # KV offload 配置
    offload_ratio: float = 0.5  # prefill 后 offload 的 token 比例
    top_k_per_head: int = 32  # decode 时每个 head 检索的 top-k token 数
    num_norm_buckets: int = 10  # 范数分桶数量
    hnsw_M: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    offload_strategy: Literal["middle", "random", "first"] = "middle"

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


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

        model_config: Qwen3Config = cast(Qwen3Config, self.model.config)
        self.num_layers = model_config.num_hidden_layers
        self.num_attention_heads = model_config.num_attention_heads
        self.num_kv_heads = getattr(
            model_config, "num_key_value_heads", self.num_attention_heads
        )
        self.head_dim = model_config.head_dim
        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size

        # 使用 HybridKVCacheManager 同时管理 GPU cache 和 CPU offload
        self.kv_cache = HybridKVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            dtype=config.dtype,
            device=self.device,
            offload_ratio=config.offload_ratio,
            top_k_per_head=config.top_k_per_head,
            num_norm_buckets=config.num_norm_buckets,
            hnsw_M=config.hnsw_M,
            hnsw_ef_construction=config.hnsw_ef_construction,
            hnsw_ef_search=config.hnsw_ef_search,
        )

        logger.info(
            f"Model loaded: {self.num_layers} layers, "
            f"{self.num_attention_heads} attention heads, "
            f"{self.num_kv_heads} KV heads, "
            f"head_dim={self.head_dim}, "
            f"hidden_size={self.hidden_size}\n"
            f"KV offload: ratio={config.offload_ratio}, "
            f"top_k_per_head={config.top_k_per_head}"
        )

    # ------------------------------------------------------------------
    # RoPE
    # ------------------------------------------------------------------

    def apply_rotary_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        """
        Args:
            q: [batch_size, num_heads, seq_len, head_dim]
            k: [batch_size, num_kv_heads, seq_len, head_dim]
            cos/sin: [batch_size, seq_len, head_dim]
        """

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # ------------------------------------------------------------------
    # 单层前向
    # ------------------------------------------------------------------

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

        # RoPE：position 从当前 cache 长度开始
        kv_seq_len = self.kv_cache.get_seq_len(layer_idx=layer_idx) + seq_len
        position_ids = torch.arange(
            kv_seq_len - seq_len,
            kv_seq_len,
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
        query_states, key_states = self.apply_rotary_emb(
            query_states, key_states, cos, sin
        )

        # ---- Prefill：将 KV 写入 GPU cache，用 FlashInfer 做 prefill attention ----
        if is_prefill:
            self.kv_cache.prefill(layer_idx, key_states, value_states)
            attn_output, _ = self.kv_cache.gpu_cache.compute_attention(
                layer_idx=layer_idx,
                query=query_states,
                is_prefill=True,
            )
            # attn_output: [1, seq_len, num_heads, head_dim]

        # ---- Decode：更新 GPU cache，做 hybrid attention（GPU local + CPU retrieved）----
        else:
            self.kv_cache.update_decode(layer_idx, key_states, value_states)

            attn_output = self.kv_cache.decode(
                layer_idx=layer_idx,
                query=query_states,
                num_q_heads=self.num_attention_heads,
            )
            # attn_output: [1, 1, num_heads, head_dim]

        # ---- Reshape 回 [batch, seq_len, hidden_size] ----
        # Prefill:  [1, seq_len, num_heads, head_dim] -> view OK（C-contiguous）
        # Decode:   [1, num_heads, 1, head_dim]       -> view OK（相同元素总数且连续）
        attn_output = attn_output.view(
            batch_size, seq_len, self.num_attention_heads * self.head_dim
        )

        # Output projection
        attn_output = attn.o_proj(attn_output)

        # Residual
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Decode step
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

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
            return torch.argmax(logits, dim=-1, keepdim=True)

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
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(-1)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

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
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)

        logger.info(f"Input: {prompt}")
        logger.info(f"Input tokens: {input_ids.shape[1]}")

        # ---- Prefill ----
        prefill_start = time.time()
        logits = self.prefill(input_ids)
        prefill_time = time.time() - prefill_start
        logger.info(f"Prefill completed in {prefill_time:.3f}s")

        # ---- KV Offload：prefill 后将中间部分 KV 卸载到 CPU 并建立分层索引 ----
        if self.config.offload_ratio > 0:
            offload_start = time.time()
            self.kv_cache.trigger_offload(strategy=self.config.offload_strategy)
            offload_time = time.time() - offload_start
            logger.info(f"KV offload completed in {offload_time:.3f}s")
            self.kv_cache.print_statistics()

        # 第一个生成的 token
        next_token_logits = logits[:, -1, :]
        next_token = self._sample_token(next_token_logits, temperature, top_p, top_k)
        generated_tokens = [next_token.item()]

        if stream:
            print(
                self.tokenizer.decode([next_token.item()], skip_special_tokens=True),
                end="",
                flush=True,
            )

        # ---- Decode loop ----
        decode_start = time.time()
        for i in range(max_new_tokens - 1):
            token_input = next_token.unsqueeze(-1)
            logits = self.decode_step(token_input)

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
            f"Decode: {len(generated_tokens)} tokens in {decode_time:.3f}s "
            f"({tokens_per_sec:.2f} tokens/s)"
        )

        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return prompt + generated_text


def main():
    config = Qwen3InferenceConfig(
        model_path="/home/wpc/huggingface/Qwen3-8B",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        offload_ratio=0.8,
        top_k_per_head=10,
        offload_strategy="middle",
    )

    logger.info("Initializing Qwen3 Inference Engine (with KV Offload)")
    engine = Qwen3Inference(config)

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
