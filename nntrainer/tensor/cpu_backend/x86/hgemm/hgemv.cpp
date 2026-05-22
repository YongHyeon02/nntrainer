// SPDX-License-Identifier: Apache-2.0
/**
 * @file   hgemv.cpp
 * @date   22 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @brief  AVX2 FP16 GEMV with FP32 accumulation
 */

#include "hgemv.h"

#include <avx2_internal.h>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace nntrainer::x86 {

void hgemv(const _FP16 *A, const _FP16 *X, _FP16 *Y, unsigned int M,
           unsigned int N, unsigned int lda, unsigned int incX,
           unsigned int incY, float alpha, float beta, bool TransA) {
  assert(incX > 0 && incY > 0);
  if (M == 0 || N == 0) {
    return;
  }

  if (TransA) {
    // Y[0..N) = beta * Y + alpha * A^T * X.
    // Stage the output in an FP32 scratch buffer so the row sweep over A
    // does not pay a FP16<->FP32 round trip on Y every iteration.
    const unsigned int out_len = N;
    std::vector<float> Y32(out_len);

    if (beta == 0.0f) {
      std::memset(Y32.data(), 0, out_len * sizeof(float));
    } else if (incY == 1) {
      const __m256 vbeta = _mm256_set1_ps(beta);
      unsigned int j = 0;
      for (; j + 8 <= out_len; j += 8) {
        __m256 y32 =
          _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + j)));
        y32 = _mm256_mul_ps(y32, vbeta);
        _mm256_storeu_ps(Y32.data() + j, y32);
      }
      for (; j < out_len; ++j) {
        Y32[j] = beta * static_cast<float>(Y[j]);
      }
    } else {
      for (unsigned int j = 0; j < out_len; ++j) {
        Y32[j] = beta * static_cast<float>(Y[j * incY]);
      }
    }

    // Per row i, Y32[0..N) += (alpha * X[i]) * A[i, 0..N).
    for (unsigned int i = 0; i < M; ++i) {
      const float scale = alpha * static_cast<float>(X[i * incX]);
      if (scale == 0.0f) {
        continue;
      }
      const __m256 vs = _mm256_set1_ps(scale);
      const _FP16 *a_row = A + i * lda;
      unsigned int j = 0;
      for (; j + 8 <= out_len; j += 8) {
        __m256 a32 =
          _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a_row + j)));
        __m256 y32 = _mm256_loadu_ps(Y32.data() + j);
        y32 = _mm256_fmadd_ps(a32, vs, y32);
        _mm256_storeu_ps(Y32.data() + j, y32);
      }
      for (; j < out_len; ++j) {
        Y32[j] += scale * static_cast<float>(a_row[j]);
      }
    }

    if (incY == 1) {
      unsigned int j = 0;
      for (; j + 8 <= out_len; j += 8) {
        __m256 y32 = _mm256_loadu_ps(Y32.data() + j);
        __m128i y16 = _mm256_cvtps_ph(y32, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(Y + j), y16);
      }
      for (; j < out_len; ++j) {
        Y[j] = static_cast<_FP16>(Y32[j]);
      }
    } else {
      for (unsigned int j = 0; j < out_len; ++j) {
        Y[j * incY] = static_cast<_FP16>(Y32[j]);
      }
    }
    return;
  }

  // TransA == false: per output row, dot product of A[i, :] with X[:].
  // The dot length (N) is the contraction; A rows are unit-stride which
  // matches the SIMD reduction pattern from sdot.
  const unsigned int dot_len = N;
  if (incX == 1) {
    const unsigned int N16 = dot_len & ~15u;
    for (unsigned int i = 0; i < M; ++i) {
      const _FP16 *a_row = A + i * lda;
      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      unsigned int j = 0;
      for (; j < N16; j += 16) {
        __m256i a_raw = _mm256_loadu_si256((const __m256i *)(a_row + j));
        __m256i x_raw = _mm256_loadu_si256((const __m256i *)(X + j));
        __m256 af0 = _mm256_cvtph_ps(_mm256_castsi256_si128(a_raw));
        __m256 af1 = _mm256_cvtph_ps(_mm256_extracti128_si256(a_raw, 1));
        __m256 xf0 = _mm256_cvtph_ps(_mm256_castsi256_si128(x_raw));
        __m256 xf1 = _mm256_cvtph_ps(_mm256_extracti128_si256(x_raw, 1));
        acc0 = _mm256_fmadd_ps(af0, xf0, acc0);
        acc1 = _mm256_fmadd_ps(af1, xf1, acc1);
      }
      __m256 acc = _mm256_add_ps(acc0, acc1);
      if (j + 8 <= dot_len) {
        __m256 a32 =
          _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a_row + j)));
        __m256 x32 =
          _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + j)));
        acc = _mm256_fmadd_ps(a32, x32, acc);
        j += 8;
      }
      float sum = avx2::internal::hsum_avx(acc);
      for (; j < dot_len; ++j) {
        sum += static_cast<float>(a_row[j]) * static_cast<float>(X[j]);
      }
      float y_new = alpha * sum;
      if (beta != 0.0f) {
        y_new += beta * static_cast<float>(Y[i * incY]);
      }
      Y[i * incY] = static_cast<_FP16>(y_new);
    }
  } else {
    for (unsigned int i = 0; i < M; ++i) {
      const _FP16 *a_row = A + i * lda;
      float sum = 0.0f;
      for (unsigned int j = 0; j < dot_len; ++j) {
        sum +=
          static_cast<float>(a_row[j]) * static_cast<float>(X[j * incX]);
      }
      float y_new = alpha * sum;
      if (beta != 0.0f) {
        y_new += beta * static_cast<float>(Y[i * incY]);
      }
      Y[i * incY] = static_cast<_FP16>(y_new);
    }
  }
}

} /* namespace nntrainer::x86 */
