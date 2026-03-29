"""
KV Cache Offload System

A hierarchical KV cache offloading system that moves middle tokens to CPU
and uses similarity-based retrieval during decode phase.

Main Components:
- HybridKVCacheManager: Main interface for managing GPU/CPU caches
- GPUKVCache: GPU-side cache management
- CPUKVCache: CPU-side storage with pin_memory
- HierarchicalIndex: Layered indexing by key norm + cosine similarity
- KVRetriever: Per-head retrieval logic
- merge_attention_lse: LSE-based attention merging

Usage:
    from kv_offload import HybridKVCacheManager

    # Initialize
    manager = HybridKVCacheManager(
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
        offload_ratio=0.5,
        top_k_per_head=32,
    )

    # Prefill phase
    for layer in range(num_layers):
        k, v = model.forward_layer(layer, ...)
        manager.prefill(layer, k, v)

    # Trigger offload
    manager.trigger_offload(strategy='middle')

    # Decode phase
    for step in range(num_decode_steps):
        for layer in range(num_layers):
            q = model.get_query(layer, ...)
            o = manager.decode(layer, q, num_q_heads)
            # Update cache with new token
            manager.update_decode(layer, new_k, new_v)
"""

from .cpu_cache import CPUKVCache
from .gpu_cache import GPUKVCache
from .hybrid_attention import merge_attention_lse
from .indexer import HierarchicalIndex
from .offload_manager import HybridKVCacheManager
from .retriever import KVRetriever

__version__ = "0.1.0"

__all__ = [
    # Main interface
    "HybridKVCacheManager",
    # Core components
    "GPUKVCache",
    "CPUKVCache",
    "HierarchicalIndex",
    "KVRetriever",
    # Attention functions
    "merge_attention_lse",
]
