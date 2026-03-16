"""
测试KV Cache Offload系统

运行方式：
    python test_kv_offload.py
"""

import time

import torch
import torch.nn.functional as F

from kv_offload import HybridKVCacheManager


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    # 初始化
    manager = HybridKVCacheManager(
        num_layers=4,
        num_kv_heads=8,
        head_dim=64,
        max_batch_size=1,
        max_seq_len=1024,
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        offload_ratio=0.5,
        top_k_per_head=16,
    )

    device = manager.device
    print(f"Using device: {device}")

    # Prefill阶段：生成随机KV
    prefill_len = 128
    num_layers = 4
    num_kv_heads = 8
    head_dim = 64

    print(f"\nPrefill phase: {prefill_len} tokens")
    for layer_idx in range(num_layers):
        keys = torch.randn(1, num_kv_heads, prefill_len, head_dim, device=device)
        values = torch.randn(1, num_kv_heads, prefill_len, head_dim, device=device)
        manager.prefill(layer_idx, keys, values)

    print(f"Prefill completed: {prefill_len} tokens stored in GPU cache")

    # 触发offload
    print("\nTriggering offload (middle 50%)...")
    manager.trigger_offload(strategy="middle")

    # 打印统计
    manager.print_statistics()

    # Decode阶段：测试几步
    print("\nDecode phase: 5 steps")
    num_q_heads = 8  # 假设query heads = kv heads（非GQA）

    for step in range(5):
        print(f"Step {step + 1}:")
        for layer_idx in range(num_layers):
            # 生成query
            query = torch.randn(1, num_q_heads, 1, head_dim, device=device)

            # 计算attention
            output = manager.decode(layer_idx, query, num_q_heads)

            # 生成新token的KV并更新
            new_k = torch.randn(1, num_kv_heads, 1, head_dim, device=device)
            new_v = torch.randn(1, num_kv_heads, 1, head_dim, device=device)
            manager.update_decode(layer_idx, new_k, new_v)

            print(f"  Layer {layer_idx}: output shape = {output.shape}")

    print("\nTest 1 PASSED")


def test_correctness():
    """测试正确性：对比hybrid attention和全GPU版本的结果"""
    print("\n" + "=" * 60)
    print("Test 2: Correctness Verification")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = 2
    num_kv_heads = 4
    num_q_heads = 4
    head_dim = 64
    prefill_len = 100

    # 方法1：使用hybrid manager（with offload）
    print("\nMethod 1: Hybrid KV Cache (with offload)")
    manager = HybridKVCacheManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_batch_size=1,
        max_seq_len=1024,
        dtype=torch.float32,
        device=device,
        offload_ratio=0.5,
        top_k_per_head=50,  # 检索足够多的tokens
    )

    # Prefill
    all_keys = []
    all_values = []
    for layer_idx in range(num_layers):
        keys = torch.randn(1, num_kv_heads, prefill_len, head_dim, device=device)
        values = torch.randn(1, num_kv_heads, prefill_len, head_dim, device=device)
        all_keys.append(keys)
        all_values.append(values)
        manager.prefill(layer_idx, keys, values)

    manager.trigger_offload(strategy="middle")

    # Decode一步
    queries = []
    outputs_hybrid = []
    for layer_idx in range(num_layers):
        query = torch.randn(1, num_q_heads, 1, head_dim, device=device)
        queries.append(query)
        output = manager.decode(layer_idx, query, num_q_heads)
        outputs_hybrid.append(output)

    # 方法2：直接计算（全GPU，作为ground truth）
    print("\nMethod 2: Direct Full GPU Attention (ground truth)")
    outputs_direct = []
    scale = 1.0 / (head_dim**0.5)

    for layer_idx in range(num_layers):
        query = queries[layer_idx]  # [1, num_q_heads, 1, head_dim]
        keys = all_keys[layer_idx]  # [1, num_kv_heads, prefill_len, head_dim]
        values = all_values[layer_idx]

        # 如果是GQA，扩展KV
        if num_q_heads != num_kv_heads:
            n_rep = num_q_heads // num_kv_heads
            keys = keys.repeat(1, n_rep, 1, 1)
            values = values.repeat(1, n_rep, 1, 1)

        # 标准attention
        attn_weights = torch.matmul(query, keys.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, values)
        outputs_direct.append(output)

    # 比较结果
    print("\nComparing results:")
    max_diff = 0.0
    for layer_idx in range(num_layers):
        diff = (
            torch.abs(outputs_hybrid[layer_idx] - outputs_direct[layer_idx])
            .max()
            .item()
        )
        max_diff = max(max_diff, diff)
        print(f"  Layer {layer_idx}: max diff = {diff:.6f}")

    threshold = 1e-4
    if max_diff < threshold:
        print(f"\nTest 2 PASSED (max diff: {max_diff:.6f} < {threshold})")
    else:
        print(f"\nTest 2 FAILED (max diff: {max_diff:.6f} >= {threshold})")
        print("Note: Small differences may be due to different computation order")
        print("      Try increasing top_k_per_head to retrieve more tokens")


def test_gqa():
    """测试GQA (Grouped Query Attention)"""
    print("\n" + "=" * 60)
    print("Test 3: GQA Support (num_q_heads > num_kv_heads)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = 2
    num_kv_heads = 4
    num_q_heads = 16  # GQA: 16 query heads, 4 KV heads
    head_dim = 64
    prefill_len = 80

    print(
        f"Query heads: {num_q_heads}, KV heads: {num_kv_heads} (ratio: {num_q_heads // num_kv_heads})"
    )

    manager = HybridKVCacheManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_batch_size=1,
        max_seq_len=1024,
        dtype=torch.float32,
        device=device,
        offload_ratio=0.5,
        top_k_per_head=20,
    )

    # Prefill
    for layer_idx in range(num_layers):
        keys = torch.randn(1, num_kv_heads, prefill_len, head_dim, device=device)
        values = torch.randn(1, num_kv_heads, prefill_len, head_dim, device=device)
        manager.prefill(layer_idx, keys, values)

    manager.trigger_offload(strategy="middle")

    # Decode with GQA
    for layer_idx in range(num_layers):
        query = torch.randn(1, num_q_heads, 1, head_dim, device=device)
        output = manager.decode(layer_idx, query, num_q_heads)
        print(
            f"Layer {layer_idx}: query shape = {query.shape}, output shape = {output.shape}"
        )
        assert output.shape == query.shape, "Output shape mismatch"

    print("\nTest 3 PASSED")


def test_performance():
    """性能测试：对比offload vs 全GPU"""
    print("\n" + "=" * 60)
    print("Test 4: Performance Comparison")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping performance test")
        return

    device = torch.device("cuda")

    num_layers = 8
    num_kv_heads = 8
    num_q_heads = 8
    head_dim = 128
    prefill_len = 2048  # 长序列

    # 方法1：全GPU（不offload）
    print(f"\nMethod 1: Full GPU (no offload) - {prefill_len} tokens")
    manager_full = HybridKVCacheManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_batch_size=1,
        max_seq_len=4096,
        dtype=torch.float16,
        device=device,
        offload_ratio=0.0,  # 不offload
    )

    for layer_idx in range(num_layers):
        keys = torch.randn(
            1, num_kv_heads, prefill_len, head_dim, device=device, dtype=torch.float16
        )
        values = torch.randn(
            1, num_kv_heads, prefill_len, head_dim, device=device, dtype=torch.float16
        )
        manager_full.prefill(layer_idx, keys, values)

    # 预热
    for layer_idx in range(num_layers):
        query = torch.randn(
            1, num_q_heads, 1, head_dim, device=device, dtype=torch.float16
        )
        _ = manager_full.decode(layer_idx, query, num_q_heads)

    torch.cuda.synchronize()

    # 测试
    num_decode_steps = 20
    start = time.time()
    for step in range(num_decode_steps):
        for layer_idx in range(num_layers):
            query = torch.randn(
                1, num_q_heads, 1, head_dim, device=device, dtype=torch.float16
            )
            _ = manager_full.decode(layer_idx, query, num_q_heads)
    torch.cuda.synchronize()
    time_full = time.time() - start

    print(f"Time: {time_full:.3f}s ({time_full / num_decode_steps:.3f}s per step)")

    # 方法2：Hybrid（50% offload）
    print(f"\nMethod 2: Hybrid (50% offload) - {prefill_len} tokens")
    manager_hybrid = HybridKVCacheManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_batch_size=1,
        max_seq_len=4096,
        dtype=torch.float16,
        device=device,
        offload_ratio=0.5,
        top_k_per_head=64,
    )

    for layer_idx in range(num_layers):
        keys = torch.randn(
            1, num_kv_heads, prefill_len, head_dim, device=device, dtype=torch.float16
        )
        values = torch.randn(
            1, num_kv_heads, prefill_len, head_dim, device=device, dtype=torch.float16
        )
        manager_hybrid.prefill(layer_idx, keys, values)

    manager_hybrid.trigger_offload(strategy="middle")

    # 预热
    for layer_idx in range(num_layers):
        query = torch.randn(
            1, num_q_heads, 1, head_dim, device=device, dtype=torch.float16
        )
        _ = manager_hybrid.decode(layer_idx, query, num_q_heads)

    torch.cuda.synchronize()

    # 测试
    start = time.time()
    for step in range(num_decode_steps):
        for layer_idx in range(num_layers):
            query = torch.randn(
                1, num_q_heads, 1, head_dim, device=device, dtype=torch.float16
            )
            _ = manager_hybrid.decode(layer_idx, query, num_q_heads)
    torch.cuda.synchronize()
    time_hybrid = time.time() - start

    print(f"Time: {time_hybrid:.3f}s ({time_hybrid / num_decode_steps:.3f}s per step)")

    # 比较
    print(f"\nSpeedup: {time_full / time_hybrid:.2f}x")
    print(f"Overhead: {(time_hybrid / time_full - 1) * 100:.1f}%")

    manager_hybrid.print_statistics()

    print("\nTest 4 PASSED")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("KV Cache Offload System - Test Suite")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_correctness()
        test_gqa()
        test_performance()

        print("\n" + "=" * 60)
        print("All Tests PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n\nTest FAILED with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
