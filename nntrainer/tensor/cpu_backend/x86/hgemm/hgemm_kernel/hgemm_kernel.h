// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_kernel.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Micro-kernel declarations for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_KERNEL_H_
#define __X86_HGEMM_KERNEL_H_

namespace nntrainer::x86 {

/**
 * @brief Primary 6x16 micro-kernel.
 *
 * Computes C[0..5][0..15] += packed_A * packed_B over @p K iterations.
 * packed_A layout: dst[k * 6 + m].
 * packed_B layout: dst[k * 16 + n].
 *
 * @param K          inner-dimension count
 * @param packed_A   FP32 packed A stripe (size K * 6)
 * @param packed_B   FP32 packed B stripe (size K * 16)
 * @param C          FP32 accumulator origin (M_tile = 6 rows, N_tile = 16 cols)
 * @param ldc        leading dimension of C (in elements)
 */
void hgemm_kernel_6x16(unsigned int K, const float *packed_A,
                       const float *packed_B, float *C, unsigned int ldc);

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMM_KERNEL_H_ */
