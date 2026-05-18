// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_util.cpp
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Utility helpers for x86 FP16 GEMM
 */

#include "hgemm_util.h"

#include <avx2_impl.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>
#include <new>

namespace nntrainer::x86 {

float *aligned_alloc_f32(std::size_t n_floats) {
  void *p = nullptr;
  std::size_t bytes = ((n_floats * sizeof(float) + 63u) / 64u) * 64u;
  if (posix_memalign(&p, 64, bytes) != 0) {
    throw std::bad_alloc();
  }
  return static_cast<float *>(p);
}

void aligned_free(float *p) { std::free(p); }

void copy_C_to_C32(const _FP16 *C, float *C32, unsigned int M, unsigned int N,
                   unsigned int c_stride, unsigned int c32_stride, float beta) {
  if (std::fpclassify(beta) == FP_ZERO) {
    const std::size_t row_bytes = static_cast<std::size_t>(N) * sizeof(float);
    for (unsigned int m = 0; m < M; ++m) {
      std::memset(C32 + m * c32_stride, 0, row_bytes);
    }
    return;
  }

  const bool beta_one = (beta == 1.0F);
  for (unsigned int m = 0; m < M; ++m) {
    float *row = C32 + m * c32_stride;
    nntrainer::avx2::vcvt_f16_f32(N, C + m * c_stride, row);
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

void copy_C32_to_C(const float *C32, _FP16 *C, unsigned int M, unsigned int N,
                   unsigned int c32_stride, unsigned int c_stride) {
  for (unsigned int m = 0; m < M; ++m) {
    nntrainer::avx2::vcvt_f32_f16(N, C32 + m * c32_stride, C + m * c_stride);
  }
}

void apply_beta_to_C(_FP16 *C, unsigned int M, unsigned int N,
                     unsigned int c_stride, float beta) {
  if (std::fpclassify(beta) == FP_ZERO) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N; ++n) {
        C[m * c_stride + n] = static_cast<_FP16>(0.0F);
      }
    }
    return;
  }

  if (beta == 1.0F) {
    return;
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * c_stride + n;
      C[idx] = static_cast<_FP16>(beta * static_cast<float>(C[idx]));
    }
  }
}

} /* namespace nntrainer::x86 */
