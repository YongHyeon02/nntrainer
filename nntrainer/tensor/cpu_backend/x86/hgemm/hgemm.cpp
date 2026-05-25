// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm.cpp
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Entry point for the x86 cache-blocked FP16 GEMM
 */

#include "hgemm.h"
#include "hgemm_common.h"
#include "hgemm_noTrans.h"
#ifdef ENABLE_TEST
#include "hgemm_test.h"
#endif
#include "hgemm_util.h"

#include <cmath>
#include <cstring>

namespace nntrainer::x86 {

namespace {

struct HgemmWorkspace {
  ~HgemmWorkspace() {
    aligned_free(pack_b);
    aligned_free(pack_a);
    aligned_free(c32);
  }

  HgemmWorkspace() = default;
  HgemmWorkspace(const HgemmWorkspace &) = delete;
  HgemmWorkspace &operator=(const HgemmWorkspace &) = delete;

  float *ensure_c32(std::size_t required) {
    return ensure_buffer(c32, c32_capacity, c32_realloc_count, required);
  }

  float *ensure_pack_a(std::size_t required) {
    return ensure_buffer(pack_a, pack_a_capacity, pack_a_realloc_count,
                         required);
  }

  float *ensure_pack_b(std::size_t required) {
    return ensure_buffer(pack_b, pack_b_capacity, pack_b_realloc_count,
                         required);
  }

  void reset_realloc_counts() {
    c32_realloc_count = 0;
    pack_a_realloc_count = 0;
    pack_b_realloc_count = 0;
  }

  float *c32 = nullptr;
  float *pack_a = nullptr;
  float *pack_b = nullptr;
  std::size_t c32_capacity = 0;
  std::size_t pack_a_capacity = 0;
  std::size_t pack_b_capacity = 0;
  std::size_t c32_realloc_count = 0;
  std::size_t pack_a_realloc_count = 0;
  std::size_t pack_b_realloc_count = 0;

private:
  static float *ensure_buffer(float *&buffer, std::size_t &capacity,
                              std::size_t &realloc_count,
                              std::size_t required) {
    if (required <= capacity) {
      return buffer;
    }

    float *next = aligned_alloc_f32(required);
    aligned_free(buffer);
    buffer = next;
    capacity = required;
    ++realloc_count;
    return buffer;
  }
};

HgemmWorkspace &get_hgemm_workspace() {
  thread_local HgemmWorkspace workspace;
  return workspace;
}

template <typename AType, typename BType, typename CType>
void hgemm_compute(bool TransA, bool TransB, unsigned int M, unsigned int N,
                   unsigned int K, float alpha, const AType *A,
                   unsigned int a_stride, const BType *B, unsigned int b_stride,
                   float beta, CType *C, unsigned int c_stride) {
  if (M == 0 || N == 0) {
    return;
  }

  if (K == 0 || std::fpclassify(alpha) == FP_ZERO) {
    apply_beta_to_C<CType>(C, M, N, c_stride, beta);
    return;
  }

  // C32 is padded on both axes so the kernel can store full 6x16 tiles even
  // when (M, N) are not multiples of (MR, NR). Tail rows / cols are written
  // with zero contributions (packed A / B at the edges is zero-padded by the
  // packing routines) and ignored on writeback.
  const unsigned int N_pad = round_up(N, X86_HGEMM_NR);
  const unsigned int M_pad = round_up(M, X86_HGEMM_MR);
  HgemmWorkspace &workspace = get_hgemm_workspace();

  float *C32 = workspace.ensure_c32(static_cast<std::size_t>(M_pad) * N_pad);
  std::memset(C32, 0, static_cast<std::size_t>(M_pad) * N_pad * sizeof(float));

  copy_C_to_C32<CType>(C, C32, M, N, c_stride, N_pad, beta);

  // Packing buffers sized for one M-block of A and one N-block of B,
  // both at the deepest K-block. Capacity uses M_pad/N_pad in case M_BLOCKING
  // or N_BLOCKING are not multiples of MR/NR (they are, but be defensive).
  const std::size_t sa_capacity =
    static_cast<std::size_t>(round_up(X86_HGEMM_M_BLOCKING, X86_HGEMM_MR)) *
    X86_HGEMM_K_BLOCKING;
  const std::size_t sb_capacity =
    static_cast<std::size_t>(X86_HGEMM_K_BLOCKING) *
    round_up(X86_HGEMM_N_BLOCKING, X86_HGEMM_NR);

  float *sa = workspace.ensure_pack_a(sa_capacity);
  float *sb = workspace.ensure_pack_b(sb_capacity);

  // Pass the *actual* (M, N) to the kernel; the packing routines cap their
  // reads at (M, N) and zero-pad up to the (MR, NR) tile boundary. Kernel
  // stores to C32 stay in-bounds because C32 is allocated at M_pad x N_pad.
  hgemm_blocked_kernel<AType, BType>(TransA, TransB, M, N, K, alpha, A,
                                     a_stride, B, b_stride, C32, N_pad, sa, sb);

  copy_C32_to_C<CType>(C32, C, M, N, N_pad, c_stride);
}

} // namespace

void hgemm(const _FP16 *A, const _FP16 *B, _FP16 *C, unsigned int M,
           unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
           unsigned int ldc, float alpha, float beta, bool TransA,
           bool TransB) {
  hgemm_compute<_FP16, _FP16, _FP16>(TransA, TransB, M, N, K, alpha, A, lda, B,
                                     ldb, beta, C, ldc);
}

void shgemm(const float *A, const _FP16 *B, float *C, unsigned int M,
            unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
            unsigned int ldc, float alpha, float beta, bool TransA,
            bool TransB) {
  hgemm_compute<float, _FP16, float>(TransA, TransB, M, N, K, alpha, A, lda, B,
                                     ldb, beta, C, ldc);
}

void hsgemm(const _FP16 *A, const float *B, float *C, unsigned int M,
            unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
            unsigned int ldc, float alpha, float beta, bool TransA,
            bool TransB) {
  hgemm_compute<_FP16, float, float>(TransA, TransB, M, N, K, alpha, A, lda, B,
                                     ldb, beta, C, ldc);
}

#ifdef ENABLE_TEST
namespace testing {

HgemmWorkspaceStats get_hgemm_workspace_stats() {
  const HgemmWorkspace &workspace = get_hgemm_workspace();
  HgemmWorkspaceStats stats;
  stats.c32_capacity = workspace.c32_capacity;
  stats.pack_a_capacity = workspace.pack_a_capacity;
  stats.pack_b_capacity = workspace.pack_b_capacity;
  stats.c32_realloc_count = workspace.c32_realloc_count;
  stats.pack_a_realloc_count = workspace.pack_a_realloc_count;
  stats.pack_b_realloc_count = workspace.pack_b_realloc_count;
  stats.total_realloc_count = stats.c32_realloc_count +
                              stats.pack_a_realloc_count +
                              stats.pack_b_realloc_count;
  stats.total_capacity_bytes =
    (stats.c32_capacity + stats.pack_a_capacity + stats.pack_b_capacity) *
    sizeof(float);
  return stats;
}

void reset_hgemm_workspace_stats() {
  get_hgemm_workspace().reset_realloc_counts();
}

} // namespace testing
#endif

} /* namespace nntrainer::x86 */
