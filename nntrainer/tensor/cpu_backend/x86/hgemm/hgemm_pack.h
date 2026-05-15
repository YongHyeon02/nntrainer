// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_pack.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Packing routines for x86 FP16 GEMM (FP16->FP32 during pack)
 */

#ifndef __X86_HGEMM_PACK_H_
#define __X86_HGEMM_PACK_H_

#include <tensor_dim.h>

namespace nntrainer::x86 {

/**
 * @brief Pack an MR-row stripe of A (FP16, row-major source) into FP32.
 *
 * Source shape: m_actual rows x k_min cols at stride lda. Output is stored
 * in k-major MR-tile layout: dst[k * MR + m] = (float) src[m * lda + k]
 * for m in [0, m_actual). Remaining rows (m_actual..MR) are filled with 0.
 *
 * @param m_actual number of valid rows (1..MR)
 * @param k_min K dimension of the stripe
 * @param src   FP16 source pointer
 * @param lda   source leading dimension (row stride in elements)
 * @param dst   FP32 packed buffer, capacity >= k_min * MR
 */
void packing_A_M6(unsigned int m_actual, unsigned int k_min,
                  const _FP16 *src, unsigned int lda, float *dst);

/**
 * @brief Pack an NR-col stripe of B (FP16, row-major source) into FP32.
 *
 * Source shape: k_min rows x n_actual cols at stride ldb. Output is stored
 * in k-major NR-tile layout: dst[k * NR + n] = (float) src[k * ldb + n]
 * for n in [0, n_actual). Remaining cols (n_actual..NR) are filled with 0.
 *
 * @param k_min    K dimension of the stripe
 * @param n_actual number of valid cols (1..NR)
 * @param src      FP16 source pointer
 * @param ldb      source leading dimension (row stride in elements)
 * @param dst      FP32 packed buffer, capacity >= k_min * NR
 */
void packing_B_N16(unsigned int k_min, unsigned int n_actual,
                   const _FP16 *src, unsigned int ldb, float *dst);

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMM_PACK_H_ */
