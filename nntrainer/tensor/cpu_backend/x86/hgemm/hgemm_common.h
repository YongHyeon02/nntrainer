// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_common.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Block-size constants for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_COMMON_H_
#define __X86_HGEMM_COMMON_H_

/// Outer blocking sizes (P3-1 initial values; re-tuned at end of P3).
/// Rationale: packed A (M_BLOCKING x K_BLOCKING x 4B = 256KB) and packed B
/// (K_BLOCKING x N_BLOCKING x 4B = 512KB) target L2-resident working sets.
#define X86_HGEMM_M_BLOCKING 256
#define X86_HGEMM_N_BLOCKING 512
#define X86_HGEMM_K_BLOCKING 256

/// Micro-kernel tile dimensions.
/// 6 x 16 = 12 YMM accumulators + 2 YMM B + 1 YMM A broadcast = 15 of 16 YMMs.
#define X86_HGEMM_MR 6
#define X86_HGEMM_NR 16

#endif /* __X86_HGEMM_COMMON_H_ */
