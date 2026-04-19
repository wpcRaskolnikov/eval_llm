import torch
import torch.nn.functional as F


def apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=1):
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


def sample_token(
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
