// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_noTrans.cpp
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Blocked dispatcher for x86 FP16 GEMM
 */

#include "hgemm_noTrans.h"
#include "hgemm_common.h"
#include "hgemm_kernel/hgemm_kernel.h"
#include "hgemm_pack.h"

#include <algorithm>

namespace nntrainer::x86 {

void hgemm_blocked_kernel(bool TransA, bool TransB, unsigned int M,
                          unsigned int N, unsigned int K, float alpha,
                          const _FP16 *A, unsigned int a_stride,
                          const _FP16 *B, unsigned int b_stride, float *C32,
                          unsigned int c32_stride, float *sa, float *sb) {
  for (unsigned int ms = 0; ms < M; ms += X86_HGEMM_M_BLOCKING) {
    unsigned int m_min = std::min<unsigned int>(M - ms, X86_HGEMM_M_BLOCKING);

    for (unsigned int ks = 0; ks < K; ks += X86_HGEMM_K_BLOCKING) {
      unsigned int k_min = std::min<unsigned int>(K - ks, X86_HGEMM_K_BLOCKING);

      // Pack a vertical stripe of A: m_min x k_min, in MR-tile blocks.
      // Layout: for each 6-row tile in order, store k_min * MR floats.
      for (unsigned int mm = 0; mm < m_min; mm += X86_HGEMM_MR) {
        unsigned int m_act =
          std::min<unsigned int>(m_min - mm, X86_HGEMM_MR);
        float *pa = sa + (mm / X86_HGEMM_MR) * (k_min * X86_HGEMM_MR);
        if (TransA) {
          packing_A_M6_trans(m_act, k_min,
                             A + ks * a_stride + (ms + mm), a_stride, alpha,
                             pa);
        } else {
          packing_A_M6(m_act, k_min, A + (ms + mm) * a_stride + ks, a_stride,
                       alpha, pa);
        }
      }

      for (unsigned int ns = 0; ns < N; ns += X86_HGEMM_N_BLOCKING) {
        unsigned int n_min =
          std::min<unsigned int>(N - ns, X86_HGEMM_N_BLOCKING);

        // Pack a horizontal stripe of B: k_min x n_min, in NR-tile blocks.
        for (unsigned int nn = 0; nn < n_min; nn += X86_HGEMM_NR) {
          unsigned int n_act =
            std::min<unsigned int>(n_min - nn, X86_HGEMM_NR);
          float *pb = sb + (nn / X86_HGEMM_NR) * (k_min * X86_HGEMM_NR);
          if (TransB) {
            packing_B_N16_trans(k_min, n_act,
                                B + (ns + nn) * b_stride + ks, b_stride, pb);
          } else {
            packing_B_N16(k_min, n_act, B + ks * b_stride + (ns + nn),
                          b_stride, pb);
          }
        }

        // Inner GEMM: iterate MR-row tiles outer, NR-col tiles inner.
        for (unsigned int mm = 0; mm < m_min; mm += X86_HGEMM_MR) {
          const float *pa =
            sa + (mm / X86_HGEMM_MR) * (k_min * X86_HGEMM_MR);
          for (unsigned int nn = 0; nn < n_min; nn += X86_HGEMM_NR) {
            const float *pb =
              sb + (nn / X86_HGEMM_NR) * (k_min * X86_HGEMM_NR);
            float *c_tile = C32 + (ms + mm) * c32_stride + (ns + nn);
            hgemm_kernel_6x16(k_min, pa, pb, c_tile, c32_stride);
          }
        }
      }
    }
  }
}

} /* namespace nntrainer::x86 */
