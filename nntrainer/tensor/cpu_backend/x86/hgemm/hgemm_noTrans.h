// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_noTrans.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Blocked NoTrans dispatcher for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_NOTRANS_H_
#define __X86_HGEMM_NOTRANS_H_

#include <tensor_dim.h>

namespace nntrainer::x86 {

/**
 * @brief Compute C32 += A * B (alpha = 1, NoTrans both) into an FP32 buffer.
 *
 * Assumes packing buffers @p sa and @p sb are sized for the configured
 * X86_HGEMM_M_BLOCKING / N_BLOCKING / K_BLOCKING, and that C32 is laid out
 * row-major with leading dimension @p ldc32.
 *
 * @param M    rows of A / C
 * @param N    cols of B / C
 * @param K    inner dimension
 * @param A    FP16 source, row-major, leading dimension lda
 * @param lda  leading dimension of A
 * @param B    FP16 source, row-major, leading dimension ldb
 * @param ldb  leading dimension of B
 * @param C32  FP32 accumulator, row-major, leading dimension ldc32
 * @param ldc32 leading dimension of C32 (>= NR-aligned upper bound of N)
 * @param sa   pre-allocated FP32 packing buffer for A (size MR * K_BLOCKING * (M_pad/MR))
 * @param sb   pre-allocated FP32 packing buffer for B (size NR * K_BLOCKING * (N_pad/NR))
 */
void hgemm_noTrans_kernel(unsigned int M, unsigned int N, unsigned int K,
                          const _FP16 *A, unsigned int lda, const _FP16 *B,
                          unsigned int ldb, float *C32, unsigned int ldc32,
                          float *sa, float *sb);

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMM_NOTRANS_H_ */
