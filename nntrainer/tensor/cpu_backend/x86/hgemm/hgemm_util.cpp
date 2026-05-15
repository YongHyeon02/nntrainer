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

#include <cstdlib>
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

} /* namespace nntrainer::x86 */
