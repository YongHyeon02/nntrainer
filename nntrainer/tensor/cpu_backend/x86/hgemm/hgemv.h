// SPDX-License-Identifier: Apache-2.0
/**
 * @file   hgemv.h
 * @date   22 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @brief  Entry point for the x86 AVX2 FP16 GEMV
 */

#ifndef __X86_HGEMV_H_
#define __X86_HGEMV_H_

#include <tensor_dim.h>

namespace nntrainer::x86 {

/**
 * @brief Compute Y = alpha * op(A) * X + beta * Y in FP16 (row-major A).
 *
 * Replaces the legacy FP16 sgemv path (whole-matrix FP16->FP32 copy + CBLAS)
 * with a direct AVX2 kernel that converts FP16->FP32 lane-by-lane via F16C
 * and accumulates in FP32 for accuracy. Two strategies:
 *   - TransA == false: per-row dot product of A[i, :] with X[:].
 *   - TransA == true : AXPY-style sweep that stages the output in an FP32
 *     scratch so Y avoids FP16<->FP32 round trips across the M rows.
 *
 * Honors the BLAS rule "do not read Y when beta == 0".
 *
 * @param A    FP16 source matrix of shape (M, N), row-major
 * @param X    FP16 input vector. Length N if !TransA else M.
 * @param Y    FP16 output vector. Length M if !TransA else N.
 * @param M    rows of A
 * @param N    cols of A
 * @param lda  leading dimension of A (row stride in elements). Must be >= N.
 * @param incX stride between consecutive X elements (>= 1)
 * @param incY stride between consecutive Y elements (>= 1)
 * @param alpha scalar applied to op(A) * X
 * @param beta  scalar applied to Y before accumulation; Y is not read when 0
 * @param TransA whether A is transposed
 */
void hgemv(const _FP16 *A, const _FP16 *X, _FP16 *Y, unsigned int M,
           unsigned int N, unsigned int lda, unsigned int incX,
           unsigned int incY, float alpha, float beta, bool TransA);

} /* namespace nntrainer::x86 */

#endif /* __X86_HGEMV_H_ */
