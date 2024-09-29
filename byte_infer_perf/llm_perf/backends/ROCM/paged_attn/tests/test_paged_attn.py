import random
import itertools
import torch
import backends.ROCM.paged_attn.paged_attn as pa                             
from typing import (List, Optional, Any, Dict, Sequence, Tuple, Union)
from backends.ROCM.paged_attn.utils import *

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
#  torch.Size([1, 8, 1, 64]) torch.Size([1, 8, 1025, 64])
# This will change depending on the compute capability.
VERSIONS = ["v1"]
# - 512 as a buffer
# MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
MAX_SEQ_LEN = (60 * 1024) // FLOAT32_BYTES - 512
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 100  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.bfloat16]
NUM_GEN_SEQS = [1]  # Arbitrary values for testing
# NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing
NUM_HEADS = [ (8, 4)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
# HEAD_SIZES = [64, 80, 96, 112, 120, 128, 192, 256]
HEAD_SIZES = [64]

BLOCK_SIZES = [32]
USE_ALIBI = [False]
KV_CACHE_DTYPE = ["auto"]
# KV_CACHE_DTYPE = ["auto", "fp8"]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out

def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: List[torch.Tensor] = []
        values_lst: List[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


def test_paged_attention(
    version: str,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    if ((kv_cache_dtype == "fp8" and head_size % 16)
            or (version == "rocm" and head_size not in (64, 128))):
        return
    print("running test:", (num_seqs, num_heads, head_size, block_size, dtype, kv_cache_dtype))
    seed_everything(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    seq_lens = [1025]
    # seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = 1025
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: List[List[int]] = []
    for _ in range(num_seqs):
        block_table = [
            0
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random(NUM_BLOCKS, block_size, 1,
                                                            num_kv_heads, head_size,
                                                            kv_cache_dtype, dtype, seed,
                                                            device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = v_scale = 1.0

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v1":
        print("shapes:",query.shape, key_cache.shape, value_cache.shape)
        pa.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    elif version in ("v2", "rocm"):
        PARTITION_SIZE = 1024 if version == "v2" else 512
        num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
        )
        max_logits = torch.empty_like(exp_sums)

        if version == "v2":
            pa.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        else:
            pa.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
    else:
        raise AssertionError(f"Unknown version: {version}")

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        alibi_slopes,
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    atol = get_default_atol(output) if is_hip() else 1e-3
    rtol = get_default_rtol(output) if is_hip() else 1e-5

    # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # so we use a relaxed tolerance for the test.
    atol, rtol = 1e-3, 1e-5
    if kv_cache_dtype == "fp8":
        atol, rtol = 1e-2, 1e-5
    # torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
    # print("    test passed!")


def test_iters():
    param_combinations = itertools.product(
                                VERSIONS,
                                NUM_GEN_SEQS,
                                NUM_HEADS,
                                HEAD_SIZES,
                                USE_ALIBI,
                                BLOCK_SIZES,
                                DTYPES,
                                KV_CACHE_DTYPE,
                                CUDA_DEVICES
                            )
    for combination in param_combinations:
        version, num_seqs, num_heads, head_size, use_alibi, \
                block_size, dtype, kv_cache_dtype, device = combination
        test_paged_attention(version=version,
                             num_seqs=num_seqs,
                             num_heads=num_heads,
                             head_size=head_size,
                             use_alibi=use_alibi,
                             block_size=block_size,
                             dtype=dtype,
                             kv_cache_dtype=kv_cache_dtype,
                             seed=0,
                             device=device)


if __name__ == "__main__":
    test_iters()