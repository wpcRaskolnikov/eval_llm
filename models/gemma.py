import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

from kv_offload import HybridKVCacheManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Gemma3InferenceConfig:
    model_path: str = "/home/wpc/huggingface/gemma-3-4b-it"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    max_batch_size: int = 1
    max_seq_len: int = 1024

    offload_ratio: float = 0.5
    top_k_per_head: int = 8
    num_norm_buckets: int = 10
    hnsw_M: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    offload_strategy: Literal["middle", "random", "first"] = "middle"

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


class Gemma3Inference:
    def __init__(self, config: Gemma3InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)

        logger.info(f"Loading model from {config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            config.model_path,
            torch_dtype=config.dtype,
            device_map=config.device,
        )
        self.model.eval()

        # Gemma3ForConditionalGeneration 结构:
        #   model.model.language_model  -> Gemma3TextModel
        #   model.lm_head               -> Linear
        self.text_model = self.model.model.language_model

        text_cfg = self.model.config.text_config
        self.num_layers = text_cfg.num_hidden_layers  # 34
        self.num_q_heads = text_cfg.num_attention_heads  # 8
        self.num_kv_heads = text_cfg.num_key_value_heads  # 4
        self.head_dim = text_cfg.head_dim  # 256
        self.hidden_size = text_cfg.hidden_size
        self.vocab_size = text_cfg.vocab_size
        self.layer_types = text_cfg.layer_types  # list[str], len=34
        self.sliding_window = text_cfg.sliding_window  # 1024
        self.final_logit_softcapping = getattr(
            text_cfg, "final_logit_softcapping", None
        )

        # 分别为 sliding / full attention 层建 KV cache manager
        self.sliding_layers = [
            i
            for i in range(self.num_layers)
            if self.layer_types[i] == "sliding_attention"
        ]
        self.full_layers = [
            i for i in range(self.num_layers) if self.layer_types[i] == "full_attention"
        ]
        self.sliding_virt = {a: v for v, a in enumerate(self.sliding_layers)}
        self.full_virt = {a: v for v, a in enumerate(self.full_layers)}

        self.sliding_kv_cache = HybridKVCacheManager(
            num_layers=len(self.sliding_layers),
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
        self.full_kv_cache = HybridKVCacheManager(
            num_layers=len(self.full_layers),
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
            f"Model loaded: {self.num_layers} layers, {self.num_q_heads} Q-heads, {self.num_kv_heads} KV-heads, head_dim={self.head_dim}\n  sliding layers: {len(self.sliding_layers)}, full layers: {len(self.full_layers)}, sliding_window={self.sliding_window}"
        )

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
        layer = self.text_model.layers[layer_idx]
        attn = layer.self_attn

        is_sliding = self.layer_types[layer_idx] == "sliding_attention"
        ltype = "sliding_attention" if is_sliding else "full_attention"
        kv_cache = self.sliding_kv_cache if is_sliding else self.full_kv_cache
        virt = self.sliding_virt[layer_idx] if is_sliding else self.full_virt[layer_idx]

        residual = hidden_states

        # Pre-attention LayerNorm
        hidden_states = layer.input_layernorm(hidden_states)

        # ---- Self-Attention ----
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape))
        # query_states: [batch, seq_len, num_q_heads, head_dim]

        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape))
        # key_states: [batch, seq_len, num_kv_heads, head_dim]

        value_states = attn.v_proj(hidden_states).view(hidden_shape)
        # value_states: [batch, seq_len, num_kv_heads, head_dim]

        # RoPE：position 从当前 cache 长度开始
        cur_len = kv_cache.get_seq_len(virt)
        position_ids = torch.arange(
            cur_len, cur_len + seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        cos, sin = self.text_model.rotary_emb(hidden_states, position_ids, ltype)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # ---- Prefill：将 KV 写入 GPU cache，用 FlashInfer 做 prefill attention ----
        if is_prefill:
            attn_output = kv_cache.prefill(virt, query_states, key_states, value_states)
            # attn_output: [1, seq_len, num_q_heads, head_dim]

        # ---- Decode：更新 GPU cache，做 hybrid attention（GPU local + CPU retrieved）----
        else:
            kv_cache.append_kv(virt, key_states, value_states)
            attn_output = kv_cache.decode(
                layer_idx=virt, query=query_states, num_q_heads=self.num_q_heads
            )
            # attn_output: [1, 1, num_q_heads, head_dim]

        # ---- Reshape 回 [batch, seq_len, hidden_size] ----
        attn_output = attn_output.view(
            batch_size, seq_len, self.num_q_heads * self.head_dim
        )

        # Output projection
        attn_output = attn.o_proj(attn_output)

        # Residual（Gemma 风格：post_attention_layernorm 作用在 attn 输出上，再加 residual）
        hidden_states = layer.post_attention_layernorm(attn_output)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        logger.info(f"Prefill: processing {seq_len} tokens")

        with torch.no_grad():
            hidden_states = self.text_model.embed_tokens(input_ids)

            for layer_idx in range(self.num_layers):
                hidden_states = self._forward_layer(
                    hidden_states, layer_idx, is_prefill=True
                )

            hidden_states = self.text_model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)
            if self.final_logit_softcapping:
                logits = (
                    torch.tanh(logits / self.final_logit_softcapping)
                    * self.final_logit_softcapping
                )

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
            hidden_states = self.text_model.embed_tokens(token_id)

            for layer_idx in range(self.num_layers):
                hidden_states = self._forward_layer(
                    hidden_states, layer_idx, is_prefill=False
                )

            hidden_states = self.text_model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)
            if self.final_logit_softcapping:
                logits = (
                    torch.tanh(logits / self.final_logit_softcapping)
                    * self.final_logit_softcapping
                )

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
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(-1)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def clear(self):
        self.sliding_kv_cache.clear()
        self.full_kv_cache.clear()

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

        self.clear()

        messages = [{"role": "user", "content": prompt}]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.device)

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
            self.sliding_kv_cache.trigger_offload(strategy=self.config.offload_strategy)
            self.full_kv_cache.trigger_offload(strategy=self.config.offload_strategy)
            offload_time = time.time() - offload_start
            logger.info(f"KV offload completed in {offload_time:.3f}s")
            self.sliding_kv_cache.print_statistics()
            self.full_kv_cache.print_statistics()

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
        eos_ids = self.tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

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

            if next_token.item() in eos_ids:
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

        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text


def main():
    config = Gemma3InferenceConfig(
        model_path="/home/wpc/huggingface/gemma-3-4b-it",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        offload_ratio=0.9,
        top_k_per_head=5,
        offload_strategy="middle",
    )

    logger.info("Initializing Gemma3 Inference Engine (with KV Offload)")
    engine = Gemma3Inference(config)

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
