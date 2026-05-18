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
#include "hgemm_util.h"

#include <cmath>
#include <cstring>

namespace nntrainer::x86 {

namespace {

void hgemm_compute(bool TransA, bool TransB, unsigned int M, unsigned int N,
                   unsigned int K, float alpha, const _FP16 *A,
                   unsigned int a_stride, const _FP16 *B,
                   unsigned int b_stride, float beta, _FP16 *C,
                   unsigned int c_stride) {
  if (M == 0 || N == 0) {
    return;
  }

  if (K == 0 || std::fpclassify(alpha) == FP_ZERO) {
    apply_beta_to_C(C, M, N, c_stride, beta);
    return;
  }

  // C32 is padded on both axes so the kernel can store full 6x16 tiles even
  // when (M, N) are not multiples of (MR, NR). Tail rows / cols are written
  // with zero contributions (packed A / B at the edges is zero-padded by the
  // packing routines) and ignored on writeback.
  const unsigned int N_pad = round_up(N, X86_HGEMM_NR);
  const unsigned int M_pad = round_up(M, X86_HGEMM_MR);

  float *C32 = aligned_alloc_f32(static_cast<std::size_t>(M_pad) * N_pad);
  std::memset(C32, 0, static_cast<std::size_t>(M_pad) * N_pad * sizeof(float));

  copy_C_to_C32(C, C32, M, N, c_stride, N_pad, beta);

  // Packing buffers sized for one M-block of A and one N-block of B,
  // both at the deepest K-block. Capacity uses M_pad/N_pad in case M_BLOCKING
  // or N_BLOCKING are not multiples of MR/NR (they are, but be defensive).
  const std::size_t sa_capacity =
    static_cast<std::size_t>(round_up(X86_HGEMM_M_BLOCKING, X86_HGEMM_MR)) *
    X86_HGEMM_K_BLOCKING;
  const std::size_t sb_capacity =
    static_cast<std::size_t>(X86_HGEMM_K_BLOCKING) *
    round_up(X86_HGEMM_N_BLOCKING, X86_HGEMM_NR);

  float *sa = aligned_alloc_f32(sa_capacity);
  float *sb = aligned_alloc_f32(sb_capacity);

  // Pass the *actual* (M, N) to the kernel; the packing routines cap their
  // reads at (M, N) and zero-pad up to the (MR, NR) tile boundary. Kernel
  // stores to C32 stay in-bounds because C32 is allocated at M_pad x N_pad.
  hgemm_blocked_kernel(TransA, TransB, M, N, K, alpha, A, a_stride, B,
                       b_stride, C32, N_pad, sa, sb);

  copy_C32_to_C(C32, C, M, N, N_pad, c_stride);

  aligned_free(sb);
  aligned_free(sa);
  aligned_free(C32);
}

} // namespace

void hgemm(const _FP16 *A, const _FP16 *B, _FP16 *C, unsigned int M,
           unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
           unsigned int ldc, float alpha, float beta, bool TransA,
           bool TransB) {
  hgemm_compute(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

} /* namespace nntrainer::x86 */
