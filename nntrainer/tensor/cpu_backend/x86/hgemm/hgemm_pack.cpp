// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_pack.cpp
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Packing routines for x86 FP16 GEMM (FP16->FP32 during pack)
 */

#include "hgemm_pack.h"
#include "hgemm_common.h"

#include <immintrin.h>

namespace nntrainer::x86 {

void packing_A_M6(unsigned int m_actual, unsigned int k_min,
                  const _FP16 *src, unsigned int src_stride, float alpha,
                  float *dst) {
  const bool alpha_one = (alpha == 1.0F);
  const __m256 valpha = _mm256_set1_ps(alpha);

  if (m_actual == X86_HGEMM_MR) {
    // Fast path: 6 rows, SIMD convert 8 k-elements per row, then transpose
    // into k-major MR-tile layout via scalar scatter store.
    unsigned int k = 0;
    alignas(32) float tmp[X86_HGEMM_MR][8];
    for (; k + 8 <= k_min; k += 8) {
      for (unsigned int mr = 0; mr < X86_HGEMM_MR; ++mr) {
        __m128i a16 = _mm_loadu_si128(
          reinterpret_cast<const __m128i *>(src + mr * src_stride + k));
        __m256 a32 = _mm256_cvtph_ps(a16);
        if (!alpha_one) {
          a32 = _mm256_mul_ps(a32, valpha);
        }
        _mm256_store_ps(tmp[mr], a32);
      }
      for (unsigned int koff = 0; koff < 8; ++koff) {
        float *out = dst + (k + koff) * X86_HGEMM_MR;
        out[0] = tmp[0][koff];
        out[1] = tmp[1][koff];
        out[2] = tmp[2][koff];
        out[3] = tmp[3][koff];
        out[4] = tmp[4][koff];
        out[5] = tmp[5][koff];
      }
    }
    for (; k < k_min; ++k) {
      float *out = dst + k * X86_HGEMM_MR;
      out[0] = alpha * static_cast<float>(src[0 * src_stride + k]);
      out[1] = alpha * static_cast<float>(src[1 * src_stride + k]);
      out[2] = alpha * static_cast<float>(src[2 * src_stride + k]);
      out[3] = alpha * static_cast<float>(src[3 * src_stride + k]);
      out[4] = alpha * static_cast<float>(src[4 * src_stride + k]);
      out[5] = alpha * static_cast<float>(src[5 * src_stride + k]);
    }
    return;
  }

  // Edge path: m_actual in [1, MR). Pad missing rows with 0.
  for (unsigned int k = 0; k < k_min; ++k) {
    float *out = dst + k * X86_HGEMM_MR;
    unsigned int m = 0;
    for (; m < m_actual; ++m) {
      out[m] = alpha * static_cast<float>(src[m * src_stride + k]);
    }
    for (; m < X86_HGEMM_MR; ++m) {
      out[m] = 0.0F;
    }
  }
}

void packing_A_M6_trans(unsigned int m_actual, unsigned int k_min,
                        const _FP16 *src, unsigned int src_stride, float alpha,
                        float *dst) {
  const bool alpha_one = (alpha == 1.0F);

  if (m_actual == X86_HGEMM_MR) {
    // Fast path: per k, m runs unit-stride over MR=6 contiguous FP16.
    // Load the first 4 via _mm_loadl_epi64 (safe 8-byte read, never reads
    // past row[3]) and finish the remaining 2 lanes in scalar. Avoids the
    // OOB risk of a full 16-byte / 8-FP16 load at the last k iteration.
    const __m128 valpha = _mm_set1_ps(alpha);
    for (unsigned int k = 0; k < k_min; ++k) {
      const _FP16 *row = src + k * src_stride;
      float *out = dst + k * X86_HGEMM_MR;
      __m128i a16 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(row));
      __m128 a32 = _mm_cvtph_ps(a16);
      if (!alpha_one) {
        a32 = _mm_mul_ps(a32, valpha);
      }
      _mm_storeu_ps(out, a32);
      out[4] = alpha * static_cast<float>(row[4]);
      out[5] = alpha * static_cast<float>(row[5]);
    }
    return;
  }

  // Edge path: m_actual in [1, MR). Pad missing rows with 0.
  for (unsigned int k = 0; k < k_min; ++k) {
    const _FP16 *row = src + k * src_stride;
    float *out = dst + k * X86_HGEMM_MR;
    unsigned int m = 0;
    for (; m < m_actual; ++m) {
      out[m] = alpha * static_cast<float>(row[m]);
    }
    for (; m < X86_HGEMM_MR; ++m) {
      out[m] = 0.0F;
    }
  }
}

void packing_B_N16(unsigned int k_min, unsigned int n_actual,
                   const _FP16 *src, unsigned int src_stride, float *dst) {
  if (n_actual == X86_HGEMM_NR) {
    // Fast path: each k-row contributes 16 contiguous FP16 -> 16 FP32.
    for (unsigned int k = 0; k < k_min; ++k) {
      const _FP16 *row = src + k * src_stride;
      __m128i b16_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row));
      __m128i b16_hi =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + 8));
      __m256 b32_lo = _mm256_cvtph_ps(b16_lo);
      __m256 b32_hi = _mm256_cvtph_ps(b16_hi);
      float *out = dst + k * X86_HGEMM_NR;
      _mm256_storeu_ps(out, b32_lo);
      _mm256_storeu_ps(out + 8, b32_hi);
    }
    return;
  }

  // Edge path: n_actual in [1, NR). Pad missing cols with 0.
  for (unsigned int k = 0; k < k_min; ++k) {
    const _FP16 *row = src + k * src_stride;
    float *out = dst + k * X86_HGEMM_NR;
    unsigned int n = 0;
    for (; n < n_actual; ++n) {
      out[n] = static_cast<float>(row[n]);
    }
    for (; n < X86_HGEMM_NR; ++n) {
      out[n] = 0.0F;
    }
  }
}

void packing_B_N16_trans(unsigned int k_min, unsigned int n_actual,
                         const _FP16 *src, unsigned int src_stride,
                         float *dst) {
  for (unsigned int k = 0; k < k_min; ++k) {
    float *out = dst + k * X86_HGEMM_NR;
    unsigned int n = 0;
    for (; n < n_actual; ++n) {
      out[n] = static_cast<float>(src[n * src_stride + k]);
    }
    for (; n < X86_HGEMM_NR; ++n) {
      out[n] = 0.0F;
    }
  }
}

} /* namespace nntrainer::x86 */
