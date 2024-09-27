#include "paged_attn_ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
}
