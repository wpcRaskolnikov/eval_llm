"""混合Attention计算：LSE合并"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def merge_attention_lse(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    合并两个attention结果（使用LSE方法，数值稳定）

    原理：
    - attention的softmax可以写成: softmax(x) = exp(x - LSE(x))
    - 两组attention的合并就是重新normalize：
      - 新的LSE = logsumexp([LSE1, LSE2])
      - 新的output = (O1 * exp(LSE1 - LSE_new) + O2 * exp(LSE2 - LSE_new))

    Args:
        o1: [batch_size, num_heads, 1, head_dim] GPU的attention输出
        lse1: [batch_size, num_heads, 1] GPU的LSE
        o2: [batch_size, num_heads, 1, head_dim] CPU的attention输出
        lse2: [batch_size, num_heads, 1] CPU的LSE
    Returns:
        o_merged: [batch_size, num_heads, 1, head_dim] 合并后的attention输出
        lse_merged: [batch_size, num_heads, 1] 合并后的LSE
    """
    # 处理-inf情况（没有有效的KV）
    # 如果lse1或lse2是-inf，说明对应的部分没有有效数据

    # 计算合并的LSE: logsumexp([lse1, lse2])
    # 数值稳定版本：max + log(sum(exp(x - max)))
    max_lse = torch.maximum(lse1, lse2)  # [batch_size, num_heads, 1]

    # 处理max_lse是-inf的情况（两个都是-inf）
    valid_mask = torch.isfinite(max_lse)

    # exp(lse - max_lse)
    exp1 = torch.exp(lse1 - max_lse)
    exp2 = torch.exp(lse2 - max_lse)

    # 处理nan（当lse是-inf时，exp(-inf - -inf) = nan）
    exp1 = torch.where(torch.isfinite(lse1), exp1, torch.zeros_like(exp1))
    exp2 = torch.where(torch.isfinite(lse2), exp2, torch.zeros_like(exp2))

    # LSE_merged = max + log(exp1 + exp2)
    lse_merged = max_lse + torch.log(exp1 + exp2)

    # 处理都是-inf的情况
    lse_merged = torch.where(
        valid_mask, lse_merged, torch.full_like(lse_merged, float("-inf"))
    )

    # 计算权重
    # weight1 = exp(lse1 - lse_merged)
    # weight2 = exp(lse2 - lse_merged)
    weight1 = torch.exp(lse1 - lse_merged).unsqueeze(-1)  # [bs, heads, 1, 1]
    weight2 = torch.exp(lse2 - lse_merged).unsqueeze(-1)

    # 处理nan
    weight1 = torch.where(torch.isfinite(weight1), weight1, torch.zeros_like(weight1))
    weight2 = torch.where(torch.isfinite(weight2), weight2, torch.zeros_like(weight2))

    # 合并输出（cast 回 o1 的 dtype，避免 float32 权重将 bfloat16/float16 输出提升为 float32）
    o_merged = (o1 * weight1 + o2 * weight2).to(o1.dtype)  # [bs, heads, 1, head_dim]

    return o_merged, lse_merged


def test_merge_correctness():
    """
    测试LSE合并的正确性：验证合并后的结果与直接计算全部KV的结果一致
    """
    import torch.nn.functional as F

    torch.manual_seed(42)
    batch_size, num_heads, head_dim = 1, 4, 64
    gpu_len, cpu_len = 50, 100

    query = torch.randn(batch_size, num_heads, 1, head_dim)
    gpu_keys = torch.randn(batch_size, num_heads, gpu_len, head_dim)
    gpu_values = torch.randn(batch_size, num_heads, gpu_len, head_dim)
    cpu_keys = torch.randn(batch_size, num_heads, cpu_len, head_dim)
    cpu_values = torch.randn(batch_size, num_heads, cpu_len, head_dim)

    scale = 1.0 / (head_dim**0.5)

    # 方法1：分别计算 GPU/CPU attention，再用 merge_attention_lse 合并
    def sdp_with_lse(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        max_s = scores.max(dim=-1, keepdim=True)[0]
        lse = (max_s + torch.log(torch.exp(scores - max_s).sum(dim=-1, keepdim=True))).squeeze(-1)
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        return out, lse

    o_gpu, lse_gpu = sdp_with_lse(query, gpu_keys, gpu_values)
    o_cpu, lse_cpu = sdp_with_lse(query, cpu_keys, cpu_values)
    output_merged, _ = merge_attention_lse(o_gpu, lse_gpu, o_cpu, lse_cpu)

    # 方法2：直接计算所有KV
    all_keys = torch.cat([gpu_keys, cpu_keys], dim=2)
    all_values = torch.cat([gpu_values, cpu_values], dim=2)
    attn_weights = torch.matmul(query, all_keys.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    output_direct = torch.matmul(attn_weights, all_values)

    # 比较
    diff = torch.abs(output_merged - output_direct).max().item()
    print(f"Max difference: {diff}")
    print(f"Test {'PASSED' if diff < 1e-5 else 'FAILED'}")

    return diff < 1e-5


if __name__ == "__main__":
    test_merge_correctness()

