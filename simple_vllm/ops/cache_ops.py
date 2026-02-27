from typing import Dict, List

import torch


def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_mapping: Dict[int, int],
) -> None:
    for src_idx, dst_idx in block_mapping.items():
        dst[dst_idx].copy_(src[src_idx], non_blocking=True)


def copy_blocks(
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    block_mapping: Dict[int, List[int]],
) -> None:
    for caches in (key_caches, value_caches):
        for cache in caches:
            for src_idx, dst_indices in block_mapping.items():
                src_block = cache[src_idx].clone()
                for dst_idx in dst_indices:
                    cache[dst_idx].copy_(src_block, non_blocking=True)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_size = key.shape
    x = key_cache.shape[-1]
    block_size = value_cache.shape[-1]

    reshaped_key = key.view(num_tokens, num_heads, head_size // x, x)
    slot_mapping = slot_mapping.to(torch.long)
    block_idx = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offset = slot_mapping % block_size

    key_cache[block_idx, :, :, block_offset, :] = reshaped_key
    value_cache[block_idx, :, :, block_offset] = value


def gather_cached_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_size = key.shape
    x = key_cache.shape[-1]
    block_size = value_cache.shape[-1]

    slot_mapping = slot_mapping.to(torch.long)
    block_idx = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offset = slot_mapping % block_size

    key_view = key.view(num_tokens, num_heads, head_size // x, x)
    key_view.copy_(key_cache[block_idx, :, :, block_offset, :])
    value.copy_(value_cache[block_idx, :, :, block_offset])
