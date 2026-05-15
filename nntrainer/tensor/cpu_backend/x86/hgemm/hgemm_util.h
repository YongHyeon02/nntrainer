// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_util.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Utility helpers for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_UTIL_H_
#define __X86_HGEMM_UTIL_H_

#include <cstddef>

namespace nntrainer::x86 {

/**
 * @brief Allocate cache-line-aligned FP32 buffer.
 * @param n_floats element count
 * @return aligned pointer, must be released with aligned_free
 */
float *aligned_alloc_f32(std::size_t n_floats);

/**
 * @brief Release a buffer allocated via aligned_alloc_f32.
 */
void aligned_free(float *p);

/**
 * @brief Round @p v up to the next multiple of @p n.
 */
inline unsigned int round_up(unsigned int v, unsigned int n) {
  return ((v + n - 1) / n) * n;
}

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMM_UTIL_H_ */
