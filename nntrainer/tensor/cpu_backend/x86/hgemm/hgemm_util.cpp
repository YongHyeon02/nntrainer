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
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <new>
#include <type_traits>

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

namespace {

/// Widen one C row into the FP32 accumulator. F16C for _FP16, plain copy for
/// float so the FP32-output GEMMs avoid a needless conversion pass.
template <typename CType>
inline void widen_C_row(const CType *src, float *dst, unsigned int N) {
  if constexpr (std::is_same_v<CType, float>) {
    std::memcpy(dst, src, static_cast<std::size_t>(N) * sizeof(float));
  } else {
    nntrainer::avx2::vcvt_f16_f32(N, src, dst);
  }
}

/// Narrow one FP32 accumulator row back into C (F16C for _FP16, copy for
/// float).
template <typename CType>
inline void narrow_C_row(const float *src, CType *dst, unsigned int N) {
  if constexpr (std::is_same_v<CType, float>) {
    std::memcpy(dst, src, static_cast<std::size_t>(N) * sizeof(float));
  } else {
    nntrainer::avx2::vcvt_f32_f16(N, src, dst);
  }
}

} // namespace

template <typename CType>
void copy_C_to_C32(const CType *C, float *C32, unsigned int M, unsigned int N,
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
    widen_C_row<CType>(C + m * c_stride, row, N);
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

template <typename CType>
void copy_C32_to_C(const float *C32, CType *C, unsigned int M, unsigned int N,
                   unsigned int c32_stride, unsigned int c_stride) {
  for (unsigned int m = 0; m < M; ++m) {
    narrow_C_row<CType>(C32 + m * c32_stride, C + m * c_stride, N);
  }
}

template <typename CType>
void apply_beta_to_C(CType *C, unsigned int M, unsigned int N,
                     unsigned int c_stride, float beta) {
  if (std::fpclassify(beta) == FP_ZERO) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N; ++n) {
        C[m * c_stride + n] = static_cast<CType>(0.0F);
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
      C[idx] = static_cast<CType>(beta * static_cast<float>(C[idx]));
    }
  }
}

template void copy_C_to_C32<_FP16>(const _FP16 *, float *, unsigned int,
                                   unsigned int, unsigned int, unsigned int,
                                   float);
template void copy_C_to_C32<float>(const float *, float *, unsigned int,
                                   unsigned int, unsigned int, unsigned int,
                                   float);
template void copy_C32_to_C<_FP16>(const float *, _FP16 *, unsigned int,
                                   unsigned int, unsigned int, unsigned int);
template void copy_C32_to_C<float>(const float *, float *, unsigned int,
                                   unsigned int, unsigned int, unsigned int);
template void apply_beta_to_C<_FP16>(_FP16 *, unsigned int, unsigned int,
                                     unsigned int, float);
template void apply_beta_to_C<float>(float *, unsigned int, unsigned int,
                                     unsigned int, float);

} /* namespace nntrainer::x86 */
