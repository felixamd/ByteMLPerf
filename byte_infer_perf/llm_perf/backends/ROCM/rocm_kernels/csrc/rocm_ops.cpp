#include "moe_ops.h"
#include "paged_attn_ops.h"
#include "cache.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &topk_softmax,
        "Apply topk softmax to the gating outputs.");
  m.def("moe_align_block_size", &moe_align_block_size,
          "Aligning the number of tokens to be processed by each expert such "
          "that it is divisible by the block size.");
  m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
  m.def("rms_norm", &rms_norm, "Apply Root Mean Square (RMS) Normalization to the input tensor.");
  m.def("fused_add_rms_norm", &fused_add_rms_norm, "In-place fused Add and RMS Normalization");
  m.def("paged_attention_rocm", &paged_attention_rocm,
        "paged_attention_rocm(Tensor! out, Tensor exp_sums,"
        "                Tensor max_logits, Tensor tmp_out,"
        "                Tensor query, Tensor key_cache,"
        "                Tensor value_cache, int num_kv_heads,"
        "                float scale, Tensor block_tables,"
        "                Tensor context_lens, int block_size,"
        "                int max_context_len,"
        "                Tensor? alibi_slopes,"
        "                str kv_cache_dtype,"
        "                float k_scale, float v_scale) -> ()"
  );
  m.def("paged_attention_v1", &paged_attention_v1,
        "paged_attention_v1("
        "    Tensor! out, Tensor query, Tensor key_cache,"
        "    Tensor value_cache, int num_kv_heads, float scale,"
        "    Tensor block_tables, Tensor seq_lens, int block_size,"
        "    int max_seq_len, Tensor? alibi_slopes,"
        "    str kv_cache_dtype, float k_scale, float v_scale,"
        "    int tp_rank, int blocksparse_local_blocks,"
        "    int blocksparse_vert_stride, int blocksparse_block_size,"
        "    int blocksparse_head_sliding_step) -> ()"
  );
  m.def("paged_attention_v2", &paged_attention_v2,
        "paged_attention_v2("
        "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
        "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
        "    Tensor value_cache, int num_kv_heads, float scale,"
        "    Tensor block_tables, Tensor seq_lens, int block_size,"
        "    int max_seq_len, Tensor? alibi_slopes,"
        "    str kv_cache_dtype, float k_scale, float v_scale,"
        "    int tp_rank, int blocksparse_local_blocks,"
        "    int blocksparse_vert_stride, int blocksparse_block_size,"
        "    int blocksparse_head_sliding_step) -> ()"
  );
  m.def("LLMM1", &LLMM1,
        "LLMM1(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) -> "
        "()"
  );
  m.def("LLMM_Silu", &LLMM_Silu,
        "LLMM_Silu(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) "
        "-> ()");
  m.def("wvSpltK", &wvSpltK,
        "wvSpltK(Tensor in_a, Tensor in_b, Tensor! out_c, int N_in,"
        "        int CuCount) -> ()");
        
  m.def("swap_blocks", &swap_blocks,
      "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  m.def("copy_blocks", &copy_blocks,
      "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
      "Tensor block_mapping) -> ()");

  m.def("reshape_and_cache", &reshape_and_cache,
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  float k_scale, float v_scale) -> ()");
  m.def("reshape_and_cache_flash", &reshape_and_cache_flash,
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        float k_scale, float v_scale) -> ()");
  m.def("convert_fp8", &convert_fp8,
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
}