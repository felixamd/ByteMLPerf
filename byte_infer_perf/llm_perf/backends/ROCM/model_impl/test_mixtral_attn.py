import torch
import random
import unittest
from modeling_mixtral import MixtralAttention, MixtralSdpaAttention
from typing import Any, Dict, List, Optional
from transformers import PretrainedConfig

class MixtralConfig(PretrainedConfig):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, max_position_embeddings, rope_theta, attention_dropout, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout


def create_input_params(mode, batch_size, seq_len):
    if mode == "context":
        input_template = {
            "is_context": True, 

            "input_ids": [[random.randint(1000, 5000) for _ in range(seq_len)] for _ in range(batch_size)], 
            "position_ids": [[i for i in range(seq_len)] for _ in range(batch_size)], 
            "attention_mask": [[1 for _ in range(seq_len)] for _ in range(batch_size)], 

            "valid_slot_ids": [i for i in range(batch_size)], 
            "all_q_len": [seq_len for _ in range(batch_size)], 
            "all_kv_len": [seq_len for _ in range(batch_size)], 

            "get_input_logits": False, 
            "override_hidden_states": True,
        }
    elif mode == "decode":
        input_template = {
            "is_context": False, 

            "input_ids": [[random.randint(1000, 5000)] for _ in range(batch_size)], 
            "position_ids": [[seq_len] for _ in range(batch_size)], 
            "attention_mask": [[1] for _ in range(batch_size)], 

            "valid_slot_ids": [i for i in range(batch_size)], 
            "all_q_len": [1 for _ in range(batch_size)], 
            "all_kv_len": [seq_len for _ in range(batch_size)], 
            
            "get_input_logits": False, 
            "override_hidden_states": True,
        }
    return input_template

# context: 
#   input_ids: [1, s_q]
#   attention_mask = [1, s_q]
#   full_attention_mask = [1, 1, s_q, s_kv] (sq == s_kv)
def get_context_masks(
    input_ids : torch.Tensor, 
    padding_mask : torch.Tensor
):
    # input_ids: [1, q_len]
    # padding_mask = [1, q_len]
    _, q_len = input_ids.shape

    # [1, q_len, q_len]
    full_attention_mask = torch.ones(
        1, q_len, q_len, 
        device=input_ids.device
    )
    full_attention_mask.tril_()
    full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask


# decode
#   input_ids: [bs, 1]
#   attention_mask = [bs, 1]
#   full_attention_mask = [bs, 1, 1, s_kv]
def get_decode_masks(
    input_ids : torch.Tensor, 
    all_kv_len: List[int]
):
    # input_ids: [batch_size, 1]
    # padding_mask: [batch_size, 1 + max_kv_len]
    batch_size, q_len = input_ids.shape
    max_qkv_len = q_len + max(all_kv_len)
    
    # [batch_size, 1, max_qkv_len]
    padding_mask = []
    for i in range(batch_size):
        cur_qkv_len = q_len + all_kv_len[i]
        mask_per_batch = [1] * cur_qkv_len + [0] * (max_qkv_len - cur_qkv_len)
        padding_mask.append(mask_per_batch)
    full_attention_mask = torch.tensor(
        padding_mask, 
        device=input_ids.device
    ).unsqueeze_(1)
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask


def build_inputs(mode, bs, seq_len, hidden_size):
    forward_inputs = create_input_params(mode, bs, seq_len)
    # list --> torch.Tensor --> cuda
    forward_inputs["input_ids"] = torch.tensor(
        forward_inputs["input_ids"]
    ).cuda()
    forward_inputs["position_ids"] = torch.tensor(
        forward_inputs["position_ids"]
    ).cuda()
    forward_inputs["attention_mask"] = torch.tensor(
        forward_inputs["attention_mask"]
    ).cuda()
    is_context = forward_inputs["is_context"]
    if is_context:
        forward_inputs["hidden_states"] = torch.rand(bs, seq_len, hidden_size).cuda().half()
        forward_inputs["full_attention_mask"] = get_context_masks(
            forward_inputs["input_ids"],
            forward_inputs["attention_mask"]
        )
    else:
        forward_inputs["hidden_states"] = torch.rand(bs, 1, hidden_size).cuda().half()
        forward_inputs["full_attention_mask"] = get_decode_masks(
            forward_inputs["input_ids"],
            forward_inputs["all_kv_len"]
        )
    forward_inputs["attention_mask"] = forward_inputs["full_attention_mask"]
    return forward_inputs

# 创建 MixtralAttention 实例
class TestMixtralAttention(unittest.TestCase):
    def test_forward_pass(self):
        config = MixtralConfig(
            hidden_size=6144,
            num_attention_heads=48,
            num_key_value_heads=8,
            max_position_embeddings=65536,
            rope_theta=1000000,
            attention_dropout=0
        )
        layer_idx = 0
        mode = "context"
        bs = 8
        seq_len = 1024
        inputs = build_inputs(mode, bs, seq_len, config.hidden_size)
        attention = MixtralAttention(config, layer_idx).cuda().half()
        attention2 = MixtralSdpaAttention(config, layer_idx).cuda().half()
        output, attn_weights, past_key_value = attention2(**inputs)
        output, attn_weights, past_key_value = attention2(**inputs)
        self.assertEqual(output.shape, (bs, seq_len, config.hidden_size))
        
        mode = "decode"
        inputs2 = build_inputs(mode, bs, seq_len, config.hidden_size)
        inputs2["past_key_value"] = past_key_value
        output, attn_weights, past_key_value = attention2(**inputs2)
        output, attn_weights, past_key_value = attention2(**inputs2)
        output, attn_weights, past_key_value = attention2(**inputs2)
        output, attn_weights, past_key_value = attention2(**inputs2)
        self.assertEqual(output.shape, (bs, 1, config.hidden_size))


if __name__ == '__main__':
    unittest.main()