// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Entry point for the x86 cache-blocked FP16 GEMM
 */

#ifndef __X86_HGEMM_H_
#define __X86_HGEMM_H_

#include <tensor_dim.h>

namespace nntrainer::x86 {

/**
 * @brief Compute C = alpha * op(A) * op(B) + beta * C in FP16.
 *
 * Replaces the legacy FP16 sgemm triple-conversion + CBLAS path with a
 * cache-blocked GEMM that converts FP16->FP32 during packing in L2-sized
 * blocks. Internal compute is done in FP32; the result is converted back to
 * FP16 on writeback.
 *
 * @param A    FP16 source matrix, row-major contiguous
 * @param B    FP16 source matrix, row-major contiguous
 * @param C    FP16 destination, row-major contiguous
 * @param M    rows of A / C
 * @param N    cols of B / C
 * @param K    inner dimension
 * @param alpha scalar applied to op(A) * op(B)
 * @param beta scalar applied to C before accumulation
 * @param TransA whether A is transposed
 * @param TransB whether B is transposed
 */
void hgemm(const _FP16 *A, const _FP16 *B, _FP16 *C, unsigned int M,
           unsigned int N, unsigned int K, float alpha, float beta, bool TransA,
           bool TransB);

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMM_H_ */
