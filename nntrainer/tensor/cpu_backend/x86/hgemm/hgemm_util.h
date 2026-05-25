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
#include <tensor_dim.h>

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
 * @brief Copy C into padded FP32 accumulation buffer with beta scaling.
 *
 * @c CType may be @c _FP16 (F16C-widened on the way in) or @c float (copied
 * directly), so FP16- and FP32-output GEMMs share one accumulator path.
 *
 * @tparam CType C element type (_FP16 or float)
 * @param C source C matrix
 * @param C32 destination FP32 matrix
 * @param M row size of C
 * @param N column size of C
 * @param c_stride row stride of C in elements
 * @param c32_stride row stride of C32 in elements
 * @param beta scale factor applied to C
 */
template <typename CType>
void copy_C_to_C32(const CType *C, float *C32, unsigned int M, unsigned int N,
                   unsigned int c_stride, unsigned int c32_stride, float beta);

/**
 * @brief Copy padded FP32 accumulation buffer back to C.
 *
 * For @c _FP16 output the result is narrowed via F16C; for @c float output it
 * is copied verbatim.
 *
 * @tparam CType C element type (_FP16 or float)
 * @param C32 source FP32 matrix
 * @param C destination C matrix
 * @param M row size of C
 * @param N column size of C
 * @param c32_stride row stride of C32 in elements
 * @param c_stride row stride of C in elements
 */
template <typename CType>
void copy_C32_to_C(const float *C32, CType *C, unsigned int M, unsigned int N,
                   unsigned int c32_stride, unsigned int c_stride);

/**
 * @brief Apply beta scaling directly to C.
 *
 * Handles the GEMM degenerate case C = beta * C when alpha is zero or K is
 * zero. Respects row stride @p c_stride and touches only logical M x N values.
 *
 * @tparam CType C element type (_FP16 or float)
 * @param C C matrix
 * @param M row size of C
 * @param N column size of C
 * @param c_stride row stride of C in elements
 * @param beta scale factor applied to C
 */
template <typename CType>
void apply_beta_to_C(CType *C, unsigned int M, unsigned int N,
                     unsigned int c_stride, float beta);

/**
 * @brief Round @p v up to the next multiple of @p n.
 */
inline unsigned int round_up(unsigned int v, unsigned int n) {
  return ((v + n - 1) / n) * n;
}

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMM_UTIL_H_ */
