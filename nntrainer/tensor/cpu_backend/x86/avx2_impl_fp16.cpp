// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_impl_fp16.cpp
 * @date   20 Feb 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source for AVX implementation
 *
 */

#include "avx2_internal.h"
#include <avx2_impl.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <util_func.h>

using namespace nntrainer::avx2::internal;

namespace nntrainer::avx2 {

void vcvt_f16_f32(unsigned int N, const _Float16 *input, float *output) {
  assert(N != 0);
  assert(input != NULL);
  assert(output != NULL);

  unsigned int idx = 0;
  const _Float16 *data = (const _Float16 *)input;

  // 16 half-precision floating point values to single-precision values
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)data));
    const __m256 vec1 =
      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(data + 8)));
    data += 16;

    _mm256_storeu_ps(output, vec0);
    _mm256_storeu_ps(output + 8, vec1);
    output += 16;
  }
  // 8 half-precision floating point values to single-precision values
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)data));
    data += 8;

    _mm256_storeu_ps(output, vec);
    output += 8;
  }
  // remaining half-precision floating point values to single-precision values
  while (idx < N) {
    *output = static_cast<float>(*data);
    ++output;
    ++data;
    ++idx;
  }
}

void vcvt_f32_f16(unsigned int N, const float *input, _Float16 *output) {
  assert(N != 0);
  assert(input != NULL);
  assert(output != NULL);

  unsigned int idx = 0;
  _Float16 *out_data = (_Float16 *)output;

  // 16 single-precision floating point values to half-precision values
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 = _mm256_loadu_ps(input);
    const __m256 vec1 = _mm256_loadu_ps(input + 8);
    input += 16;

    _mm_storeu_si128((__m128i *)out_data,
                     _mm256_cvtps_ph(vec0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i *)(out_data + 8),
                     _mm256_cvtps_ph(vec1, _MM_FROUND_TO_NEAREST_INT));
    out_data += 16;
  }
  // 8 single-precision floating point values to half-precision values
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_loadu_ps(input);
    input += 8;

    _mm_storeu_si128((__m128i *)out_data,
                     _mm256_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT));
    out_data += 8;
  }
  // 4 single-precision floating point values to half-precision values
  for (; N - idx >= 4; idx += 4) {
    const __m128 vec = _mm_loadu_ps(input);
    input += 4;

    _mm_storeu_si64((__m128i *)out_data,
                    _mm_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT));
    out_data += 4;
  }
  // remaining single-precision floating point values to half-precision values
  while (idx < N) {
    *out_data = static_cast<_Float16>(*input);
    ++out_data;
    ++input;
    ++idx;
  }
}

bool is_valid(const unsigned int N, const _Float16 *input) {
  assert(N != 0);
  assert(input != NULL);

  int temp = 0;
  unsigned int idx = 0;

  const __m256 SIGN_MASK = _mm256_set1_ps(-0.0);
  const __m256 INF = _mm256_set1_ps(std::numeric_limits<float>::infinity());

  // 16 single-precision check : ( X != X )
  for (; N - idx >= 16; idx += 16) {
    __m256 vec0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)input));
    __m256 vec1 =
      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(input + 8)));

    input += 16;

    // check NaN in vec0
    __m256 res = _mm256_cmp_ps(vec0, vec0, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);
    if (temp)
      return false;

    // check infinity in vec0
    vec0 = _mm256_andnot_ps(SIGN_MASK, vec0);
    vec0 = _mm256_cmp_ps(vec0, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec0);
    if (temp)
      return false;

    // check NaN in vec1
    __m256 res1 = _mm256_cmp_ps(vec1, vec1, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res1);

    if (temp)
      return false;

    // check infinity in vec1
    vec1 = _mm256_andnot_ps(SIGN_MASK, vec1);
    vec1 = _mm256_cmp_ps(vec1, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec1);

    if (temp)
      return false;
  }

  // 8 single-precision check : ( X != X )
  for (; N - idx >= 8; idx += 8) {
    __m256 vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)input));
    input += 8;
    __m256 res = _mm256_cmp_ps(vec, vec, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return false;

    // check infinity in vec1
    vec = _mm256_andnot_ps(SIGN_MASK, vec);
    vec = _mm256_cmp_ps(vec, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec);

    if (temp)
      return false;
  }

  while (idx < N) {
    if (!isFloatValid(*input)) {
      return false;
    }
    ++input;
    ++idx;
  }

  return true;
}

// ============================================================
// FP16 elementwise operations
// ============================================================

void ele_mul(const unsigned int N, const _Float16 *X, const _Float16 *Y,
             _Float16 *Z, float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int i = 0;
    if (i_stride == 0) {
      float y0_f32 = static_cast<float>(Y[0]);
      __m256 vy = _mm256_set1_ps(y0_f32);
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_mul_ps(x_lo, vy);
        __m256 z_hi = _mm256_mul_ps(x_hi, vy);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_mul_ps(x, vy);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) * y0_f32);
      }
    } else if (i_stride == 1) {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_mul_ps(x_lo, y_lo);
        __m256 z_hi = _mm256_mul_ps(x_hi, y_hi);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_mul_ps(x, y);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) *
                                     static_cast<float>(Y[i]));
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) *
                                     static_cast<float>(Y[i * i_stride]));
      }
    }
  } else if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    unsigned int i = 0;

    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_mul_ps(_mm256_mul_ps(x_lo, vy), alpha_v);
        __m256 z_hi = _mm256_mul_ps(_mm256_mul_ps(x_hi, vy), alpha_v);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_mul_ps(_mm256_mul_ps(x, vy), alpha_v);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    } else {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_mul_ps(_mm256_mul_ps(x_lo, y_lo), alpha_v);
        __m256 z_hi = _mm256_mul_ps(_mm256_mul_ps(x_hi, y_hi), alpha_v);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_mul_ps(_mm256_mul_ps(x, y), alpha_v);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    }
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      float yf = static_cast<float>(Y[i * i_stride]);
      float zf = xf * alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(Z[i]));
      Z[i] = static_cast<_Float16>(zf);
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(*X);
      float yf = static_cast<float>(*Y);
      float zf = xf * alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(*Z));
      *Z = static_cast<_Float16>(zf);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

void ele_add(const unsigned int N, const _Float16 *X, const _Float16 *Y,
             _Float16 *Z, float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int i = 0;
    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_add_ps(x_lo, vy);
        __m256 z_hi = _mm256_add_ps(x_hi, vy);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_add_ps(x, vy);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) +
                                     static_cast<float>(Y[0]));
      }
    } else if (i_stride == 1) {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_add_ps(x_lo, y_lo);
        __m256 z_hi = _mm256_add_ps(x_hi, y_hi);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_add_ps(x, y);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) +
                                     static_cast<float>(Y[i]));
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) +
                                     static_cast<float>(Y[i * i_stride]));
      }
    }
  } else if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    unsigned int i = 0;

    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_fmadd_ps(alpha_v, vy, x_lo);
        __m256 z_hi = _mm256_fmadd_ps(alpha_v, vy, x_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_fmadd_ps(alpha_v, vy, x);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    } else {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_fmadd_ps(alpha_v, y_lo, x_lo);
        __m256 z_hi = _mm256_fmadd_ps(alpha_v, y_hi, x_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_fmadd_ps(alpha_v, y, x);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    }
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      float yf = static_cast<float>(Y[i * i_stride]);
      float zf = xf + alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(Z[i]));
      Z[i] = static_cast<_Float16>(zf);
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(*X);
      float yf = static_cast<float>(*Y);
      float zf = xf + alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(*Z));
      *Z = static_cast<_Float16>(zf);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

void ele_sub(const unsigned int N, const _Float16 *X, const _Float16 *Y,
             _Float16 *Z, float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int i = 0;
    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_sub_ps(x_lo, vy);
        __m256 z_hi = _mm256_sub_ps(x_hi, vy);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_sub_ps(x, vy);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) -
                                     static_cast<float>(Y[0]));
      }
    } else if (i_stride == 1) {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_sub_ps(x_lo, y_lo);
        __m256 z_hi = _mm256_sub_ps(x_hi, y_hi);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_sub_ps(x, y);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) -
                                     static_cast<float>(Y[i]));
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) -
                                     static_cast<float>(Y[i * i_stride]));
      }
    }
  } else if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    unsigned int i = 0;

    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_fnmadd_ps(alpha_v, vy, x_lo);
        __m256 z_hi = _mm256_fnmadd_ps(alpha_v, vy, x_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_fnmadd_ps(alpha_v, vy, x);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    } else {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_fnmadd_ps(alpha_v, y_lo, x_lo);
        __m256 z_hi = _mm256_fnmadd_ps(alpha_v, y_hi, x_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_fnmadd_ps(alpha_v, y, x);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    }
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      float yf = static_cast<float>(Y[i * i_stride]);
      float zf = xf - alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(Z[i]));
      Z[i] = static_cast<_Float16>(zf);
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(*X);
      float yf = static_cast<float>(*Y);
      float zf = xf - alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(*Z));
      *Z = static_cast<_Float16>(zf);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

void ele_div(const unsigned int N, const _Float16 *X, const _Float16 *Y,
             _Float16 *Z, float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  // Newton-Raphson reciprocal helper constant
  const __m256 two = _mm256_set1_ps(2.0f);

  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int i = 0;
    if (i_stride == 0) {
      // Precompute refined reciprocal of Y[0] once (rcp + Newton-Raphson)
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      __m256 rcp_est = _mm256_rcp_ps(vy);
      __m256 vy_inv =
        _mm256_mul_ps(rcp_est, _mm256_fnmadd_ps(vy, rcp_est, two));

      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_mul_ps(x_lo, vy_inv);
        __m256 z_hi = _mm256_mul_ps(x_hi, vy_inv);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_mul_ps(x, vy_inv);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) /
                                     static_cast<float>(Y[0]));
      }
    } else if (i_stride == 1) {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        // rcp + Newton-Raphson for per-element reciprocal
        __m256 rcp_lo = _mm256_rcp_ps(y_lo);
        __m256 inv_lo =
          _mm256_mul_ps(rcp_lo, _mm256_fnmadd_ps(y_lo, rcp_lo, two));
        __m256 rcp_hi = _mm256_rcp_ps(y_hi);
        __m256 inv_hi =
          _mm256_mul_ps(rcp_hi, _mm256_fnmadd_ps(y_hi, rcp_hi, two));
        __m256 z_lo = _mm256_mul_ps(x_lo, inv_lo);
        __m256 z_hi = _mm256_mul_ps(x_hi, inv_hi);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 rcp_y = _mm256_rcp_ps(y);
        __m256 inv_y = _mm256_mul_ps(rcp_y, _mm256_fnmadd_ps(y, rcp_y, two));
        __m256 z = _mm256_mul_ps(x, inv_y);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) /
                                     static_cast<float>(Y[i]));
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) /
                                     static_cast<float>(Y[i * i_stride]));
      }
    }
  } else if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    unsigned int i = 0;

    if (i_stride == 0) {
      // Precompute refined reciprocal of (alpha * Y[0])
      __m256 denom =
        _mm256_mul_ps(alpha_v, _mm256_set1_ps(static_cast<float>(Y[0])));
      __m256 rcp_d = _mm256_rcp_ps(denom);
      __m256 inv_d = _mm256_mul_ps(rcp_d, _mm256_fnmadd_ps(denom, rcp_d, two));

      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_mul_ps(x_lo, inv_d);
        __m256 z_hi = _mm256_mul_ps(x_hi, inv_d);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_mul_ps(x, inv_d);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    } else {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 d_lo = _mm256_mul_ps(alpha_v, y_lo);
        __m256 d_hi = _mm256_mul_ps(alpha_v, y_hi);
        // rcp + Newton-Raphson for per-element reciprocal
        __m256 rcp_lo = _mm256_rcp_ps(d_lo);
        __m256 inv_lo =
          _mm256_mul_ps(rcp_lo, _mm256_fnmadd_ps(d_lo, rcp_lo, two));
        __m256 rcp_hi = _mm256_rcp_ps(d_hi);
        __m256 inv_hi =
          _mm256_mul_ps(rcp_hi, _mm256_fnmadd_ps(d_hi, rcp_hi, two));
        __m256 z_lo = _mm256_mul_ps(x_lo, inv_lo);
        __m256 z_hi = _mm256_mul_ps(x_hi, inv_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 d = _mm256_mul_ps(alpha_v, y);
        __m256 rcp_d = _mm256_rcp_ps(d);
        __m256 inv_d = _mm256_mul_ps(rcp_d, _mm256_fnmadd_ps(d, rcp_d, two));
        __m256 z = _mm256_mul_ps(x, inv_d);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    }
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      float yf = static_cast<float>(Y[i * i_stride]);
      float zf = xf / (alpha * yf) +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(Z[i]));
      Z[i] = static_cast<_Float16>(zf);
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(*X);
      float yf = static_cast<float>(*Y);
      float zf = xf / (alpha * yf) +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(*Z));
      *Z = static_cast<_Float16>(zf);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

} // namespace nntrainer::avx2
