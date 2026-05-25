// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_pack.cpp
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Packing routines for x86 FP16 GEMM (source->FP32 during pack)
 */

#include "hgemm_pack.h"
#include "hgemm_common.h"

#include <immintrin.h>
#include <type_traits>

namespace nntrainer::x86 {

namespace {

/**
 * @brief Load 8 contiguous source elements and widen to FP32.
 *
 * For @c _FP16 sources this is an F16C convert; for @c float it is a plain
 * 256-bit load. Reads 8 elements starting at @p p.
 */
template <typename SrcT> inline __m256 load8_to_f32(const SrcT *p) {
  if constexpr (std::is_same_v<SrcT, float>) {
    return _mm256_loadu_ps(p);
  } else {
    return _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
  }
}

/**
 * @brief Load 4 contiguous source elements and widen to FP32.
 *
 * Used by the transposed-A fast path, which must not over-read past the MR=6
 * contiguous lanes at the final k. For @c _FP16 the 8-byte @c _mm_loadl_epi64
 * never reads past lane 3; for @c float a 16-byte load covers lanes 0..3.
 */
template <typename SrcT> inline __m128 load4_to_f32(const SrcT *p) {
  if constexpr (std::is_same_v<SrcT, float>) {
    return _mm_loadu_ps(p);
  } else {
    return _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(p)));
  }
}

} // namespace

template <typename SrcT>
void packing_A_M6(unsigned int m_actual, unsigned int k_min, const SrcT *src,
                  unsigned int src_stride, float alpha, float *dst) {
  const bool alpha_one = (alpha == 1.0F);
  const __m256 valpha = _mm256_set1_ps(alpha);

  if (m_actual == X86_HGEMM_MR) {
    // Fast path: 6 rows, SIMD convert 8 k-elements per row, then transpose
    // into k-major MR-tile layout via scalar scatter store.
    unsigned int k = 0;
    alignas(32) float tmp[X86_HGEMM_MR][8];
    for (; k + 8 <= k_min; k += 8) {
      for (unsigned int mr = 0; mr < X86_HGEMM_MR; ++mr) {
        __m256 a32 = load8_to_f32<SrcT>(src + mr * src_stride + k);
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

template <typename SrcT>
void packing_A_M6_trans(unsigned int m_actual, unsigned int k_min,
                        const SrcT *src, unsigned int src_stride, float alpha,
                        float *dst) {
  const bool alpha_one = (alpha == 1.0F);

  if (m_actual == X86_HGEMM_MR) {
    // Fast path: per k, m runs unit-stride over MR=6 contiguous elements.
    // Load the first 4 lanes wide (never reads past row[3]) and finish the
    // remaining 2 lanes in scalar. Avoids the OOB risk of a full 8-element
    // load at the last k iteration.
    const __m128 valpha = _mm_set1_ps(alpha);
    for (unsigned int k = 0; k < k_min; ++k) {
      const SrcT *row = src + k * src_stride;
      float *out = dst + k * X86_HGEMM_MR;
      __m128 a32 = load4_to_f32<SrcT>(row);
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
    const SrcT *row = src + k * src_stride;
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

template <typename SrcT>
void packing_B_N16(unsigned int k_min, unsigned int n_actual, const SrcT *src,
                   unsigned int src_stride, float *dst) {
  if (n_actual == X86_HGEMM_NR) {
    // Fast path: each k-row contributes 16 contiguous elements -> 16 FP32.
    for (unsigned int k = 0; k < k_min; ++k) {
      const SrcT *row = src + k * src_stride;
      __m256 b32_lo = load8_to_f32<SrcT>(row);
      __m256 b32_hi = load8_to_f32<SrcT>(row + 8);
      float *out = dst + k * X86_HGEMM_NR;
      _mm256_storeu_ps(out, b32_lo);
      _mm256_storeu_ps(out + 8, b32_hi);
    }
    return;
  }

  // Edge path: n_actual in [1, NR). Pad missing cols with 0.
  for (unsigned int k = 0; k < k_min; ++k) {
    const SrcT *row = src + k * src_stride;
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

template <typename SrcT>
void packing_B_N16_trans(unsigned int k_min, unsigned int n_actual,
                         const SrcT *src, unsigned int src_stride, float *dst) {
  if (n_actual == X86_HGEMM_NR) {
    // Fast path: per n-row, k runs unit-stride. SIMD-convert 8 contiguous
    // k-elements per row, then transpose into k-major NR-tile layout via
    // scalar scatter store. Mirrors packing_A_M6 fast path.
    unsigned int k = 0;
    alignas(32) float tmp[X86_HGEMM_NR][8];
    for (; k + 8 <= k_min; k += 8) {
      for (unsigned int nr = 0; nr < X86_HGEMM_NR; ++nr) {
        __m256 b32 = load8_to_f32<SrcT>(src + nr * src_stride + k);
        _mm256_store_ps(tmp[nr], b32);
      }
      for (unsigned int koff = 0; koff < 8; ++koff) {
        float *out = dst + (k + koff) * X86_HGEMM_NR;
        for (unsigned int nr = 0; nr < X86_HGEMM_NR; ++nr) {
          out[nr] = tmp[nr][koff];
        }
      }
    }
    for (; k < k_min; ++k) {
      float *out = dst + k * X86_HGEMM_NR;
      for (unsigned int n = 0; n < X86_HGEMM_NR; ++n) {
        out[n] = static_cast<float>(src[n * src_stride + k]);
      }
    }
    return;
  }

  // Edge path: n_actual in [1, NR). Pad missing cols with 0.
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

// Explicit instantiations: _FP16 source (pure-FP16 / hsgemm A / shgemm B) and
// float source (mixed-precision FP32 operand).
template void packing_A_M6<_FP16>(unsigned int, unsigned int, const _FP16 *,
                                  unsigned int, float, float *);
template void packing_A_M6<float>(unsigned int, unsigned int, const float *,
                                  unsigned int, float, float *);
template void packing_A_M6_trans<_FP16>(unsigned int, unsigned int,
                                        const _FP16 *, unsigned int, float,
                                        float *);
template void packing_A_M6_trans<float>(unsigned int, unsigned int,
                                        const float *, unsigned int, float,
                                        float *);
template void packing_B_N16<_FP16>(unsigned int, unsigned int, const _FP16 *,
                                   unsigned int, float *);
template void packing_B_N16<float>(unsigned int, unsigned int, const float *,
                                   unsigned int, float *);
template void packing_B_N16_trans<_FP16>(unsigned int, unsigned int,
                                         const _FP16 *, unsigned int, float *);
template void packing_B_N16_trans<float>(unsigned int, unsigned int,
                                         const float *, unsigned int, float *);

} /* namespace nntrainer::x86 */
