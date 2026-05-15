// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm.cpp
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Entry point for the x86 cache-blocked FP16 GEMM (P3-1: NoTrans only)
 */

#include "hgemm.h"
#include "hgemm_common.h"
#include "hgemm_noTrans.h"
#include "hgemm_util.h"

#include <avx2_impl.h>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace nntrainer::x86 {

namespace {

inline unsigned int round_up_to(unsigned int v, unsigned int n) {
  return ((v + n - 1) / n) * n;
}

/// Seed the [0..M, 0..N] window of C32 with beta * C (FP16->FP32). Caller
/// must have pre-zeroed the full padded C32 buffer; pad rows / cols stay 0.
void seed_C32(unsigned int M, unsigned int N, unsigned int N_pad,
              const _FP16 *C, unsigned int ldc, float beta, float *C32) {
  if (std::fpclassify(beta) == FP_ZERO) {
    return;
  }

  const bool beta_one = (beta == 1.0F);
  for (unsigned int m = 0; m < M; ++m) {
    float *row = C32 + m * N_pad;
    nntrainer::avx2::vcvt_f16_f32(N, C + m * ldc, row);
    if (!beta_one) {
      const __m256 vbeta = _mm256_set1_ps(beta);
      unsigned int n = 0;
      for (; n + 8 <= N; n += 8) {
        __m256 v = _mm256_loadu_ps(row + n);
        _mm256_storeu_ps(row + n, _mm256_mul_ps(vbeta, v));
      }
      for (; n < N; ++n) {
        row[n] *= beta;
      }
    }
  }
}

/// Convert the [0..M, 0..N] window of C32 back to FP16 in C.
void writeback_C32(unsigned int M, unsigned int N, unsigned int N_pad,
                   const float *C32, _FP16 *C, unsigned int ldc) {
  for (unsigned int m = 0; m < M; ++m) {
    nntrainer::avx2::vcvt_f32_f16(N, C32 + m * N_pad, C + m * ldc);
  }
}

} // namespace

void hgemm_fp16_noTrans(unsigned int M, unsigned int N, unsigned int K,
                        const _FP16 *A, unsigned int lda, const _FP16 *B,
                        unsigned int ldb, float beta, _FP16 *C,
                        unsigned int ldc) {
  // C32 is padded on both axes so the kernel can store full 6x16 tiles even
  // when (M, N) are not multiples of (MR, NR). Tail rows / cols are written
  // with zero contributions (packed A / B at the edges is zero-padded inside
  // packing_A_M6 / packing_B_N16) and ignored on writeback.
  const unsigned int N_pad = round_up_to(N, X86_HGEMM_NR);
  const unsigned int M_pad = round_up_to(M, X86_HGEMM_MR);

  float *C32 = aligned_alloc_f32(static_cast<std::size_t>(M_pad) * N_pad);
  std::memset(C32, 0, static_cast<std::size_t>(M_pad) * N_pad * sizeof(float));

  seed_C32(M, N, N_pad, C, ldc, beta, C32);

  // Packing buffers sized for one M-block of A and one N-block of B,
  // both at the deepest K-block. Capacity uses M_pad/N_pad in case M_BLOCKING
  // or N_BLOCKING are not multiples of MR/NR (they are, but be defensive).
  const std::size_t sa_capacity =
    static_cast<std::size_t>(round_up_to(X86_HGEMM_M_BLOCKING, X86_HGEMM_MR)) *
    X86_HGEMM_K_BLOCKING;
  const std::size_t sb_capacity =
    static_cast<std::size_t>(X86_HGEMM_K_BLOCKING) *
    round_up_to(X86_HGEMM_N_BLOCKING, X86_HGEMM_NR);

  float *sa = aligned_alloc_f32(sa_capacity);
  float *sb = aligned_alloc_f32(sb_capacity);

  // Pass the *actual* (M, N) to the kernel; the packing routines cap their
  // reads at (M, N) and zero-pad up to the (MR, NR) tile boundary. Kernel
  // stores to C32 stay in-bounds because C32 is allocated at M_pad x N_pad.
  hgemm_noTrans_kernel(M, N, K, A, lda, B, ldb, C32, N_pad, sa, sb);

  writeback_C32(M, N, N_pad, C32, C, ldc);

  aligned_free(sb);
  aligned_free(sa);
  aligned_free(C32);
}

} /* namespace nntrainer::x86 */
