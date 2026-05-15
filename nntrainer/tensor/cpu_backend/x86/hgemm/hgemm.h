// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Entry point for the x86 cache-blocked FP16 GEMM (P3-1: NoTrans only)
 */

#ifndef __X86_HGEMM_H_
#define __X86_HGEMM_H_

#include <tensor_dim.h>

namespace nntrainer::x86 {

/**
 * @brief Compute C = A * B + beta * C in FP16, NoTrans both, alpha = 1.
 *
 * Replaces the legacy FP16 sgemm triple-conversion + CBLAS path with a
 * cache-blocked GEMM that converts FP16->FP32 during packing in L2-sized
 * blocks. Internal compute is done in FP32; the result is converted back to
 * FP16 on writeback.
 *
 * @param M    rows of A / C
 * @param N    cols of B / C
 * @param K    inner dimension
 * @param A    FP16 source matrix, row-major, leading dimension lda
 * @param lda  leading dimension of A (>= K)
 * @param B    FP16 source matrix, row-major, leading dimension ldb
 * @param ldb  leading dimension of B (>= N)
 * @param beta scalar applied to C before accumulation
 * @param C    FP16 destination, row-major, leading dimension ldc
 * @param ldc  leading dimension of C (>= N)
 */
void hgemm_fp16_noTrans(unsigned int M, unsigned int N, unsigned int K,
                        const _FP16 *A, unsigned int lda, const _FP16 *B,
                        unsigned int ldb, float beta, _FP16 *C,
                        unsigned int ldc);

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMM_H_ */
