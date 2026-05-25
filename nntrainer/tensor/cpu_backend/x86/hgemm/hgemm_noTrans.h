// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_noTrans.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Blocked dispatcher for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_NOTRANS_H_
#define __X86_HGEMM_NOTRANS_H_

#include <tensor_dim.h>

namespace nntrainer::x86 {

/**
 * @brief Compute C32 += alpha * op(A) * op(B) into an FP32 buffer.
 *
 * Assumes packing buffers @p sa and @p sb are sized for the configured
 * X86_HGEMM_M_BLOCKING / N_BLOCKING / K_BLOCKING, and that C32 is laid out
 * row-major with stride @p c32_stride. @c AType / @c BType select whether each
 * operand is widened from @c _FP16 or re-laid out from @c float while packing,
 * so the same loop drives both the FP16 and mixed-precision GEMMs.
 *
 * @tparam AType A element type (_FP16 or float)
 * @tparam BType B element type (_FP16 or float)
 * @param TransA whether A is transposed
 * @param TransB whether B is transposed
 * @param M    rows of A / C
 * @param N    cols of B / C
 * @param K    inner dimension
 * @param alpha scalar applied to op(A) * op(B)
 * @param A    source, row-major
 * @param a_stride row stride of A in elements
 * @param B    source, row-major
 * @param b_stride row stride of B in elements
 * @param C32  FP32 accumulator, row-major
 * @param c32_stride row stride of C32 in elements
 * @param sa   pre-allocated FP32 packing buffer for A (size MR * K_BLOCKING *
 * (M_pad/MR))
 * @param sb   pre-allocated FP32 packing buffer for B (size NR * K_BLOCKING *
 * (N_pad/NR))
 */
template <typename AType, typename BType>
void hgemm_blocked_kernel(bool TransA, bool TransB, unsigned int M,
                          unsigned int N, unsigned int K, float alpha,
                          const AType *A, unsigned int a_stride, const BType *B,
                          unsigned int b_stride, float *C32,
                          unsigned int c32_stride, float *sa, float *sb);

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMM_NOTRANS_H_ */
