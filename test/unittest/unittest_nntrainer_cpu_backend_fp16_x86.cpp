// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend_fp16_x86.cpp
 * @date	16 June 2026
 * @brief	x86-only unit tests for the cache-blocked FP16 GEMM/GEMV backend
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 * @note Split out of unittest_nntrainer_cpu_backend_fp16.cpp (which is built
 * for Android only) so the x86 cache-blocked GEMM/GEMV tests build under meson
 * without touching that file. The whole body is gated on x86, so the target is
 * harmless to build on other architectures.
 */

#include "nntrainer_test_util.h"

#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(static_cast<float>(
                                      src[i * N + j] - src2[i * N + j])));
      err_sum += std::abs(static_cast<float>(src[i * N + j] - src2[i * N + j]));
    }
  }
  return max_diff;
}

/// FP16 sgemv path: exercises the x86 AVX2 hgemv kernel.
/// Reference is a local FP32 GEMV loop to avoid using the optimized backend
/// as the oracle for this backend test.
static void run_sgemv_fp16_hgemv_test(unsigned int M, unsigned int N,
                                      bool TransA = false, float alpha = 1.0F,
                                      float beta = 0.0F,
                                      unsigned int lda_extra = 0,
                                      unsigned int incX = 1,
                                      unsigned int incY = 1) {
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, N + lda_extra);
  const unsigned int lenX = TransA ? M : N;
  const unsigned int lenY = TransA ? N : M;

  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * lda);
  const std::size_t x_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenX) * std::max(incX, 1u));
  const std::size_t y_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenY) * std::max(incY, 1u));

  auto A_fp16 = generate_random_vector<_FP16>(a_size, -0.25F, 0.25F);
  auto X_fp16 = generate_random_vector<_FP16>(x_size, -0.25F, 0.25F);
  auto Y_fp16 = generate_random_vector<_FP16>(y_size, -0.25F, 0.25F);
  auto Y_before = Y_fp16;

  std::vector<float> Y_fp32_ref(y_size);
  for (std::size_t i = 0; i < y_size; ++i) {
    Y_fp32_ref[i] = static_cast<float>(Y_before[i]);
  }

  if (M != 0 && N != 0) {
    if (!TransA) {
      for (unsigned int i = 0; i < M; ++i) {
        float acc = 0.0F;
        for (unsigned int j = 0; j < N; ++j) {
          acc += static_cast<float>(A_fp16[i * lda + j]) *
                 static_cast<float>(X_fp16[j * incX]);
        }
        const std::size_t y_idx = static_cast<std::size_t>(i) * incY;
        Y_fp32_ref[y_idx] = alpha * acc + beta * Y_fp32_ref[y_idx];
      }
    } else {
      for (unsigned int j = 0; j < N; ++j) {
        float acc = 0.0F;
        for (unsigned int i = 0; i < M; ++i) {
          acc += static_cast<float>(A_fp16[i * lda + j]) *
                 static_cast<float>(X_fp16[i * incX]);
        }
        const std::size_t y_idx = static_cast<std::size_t>(j) * incY;
        Y_fp32_ref[y_idx] = alpha * acc + beta * Y_fp32_ref[y_idx];
      }
    }
  }

  // System under test: FP16 sgemv, routed to x86::hgemv by the x86 backend
  // dispatcher for row-major inputs.
  nntrainer::sgemv(0, TransA, M, N, alpha, A_fp16.data(), lda, X_fp16.data(),
                   incX, beta, Y_fp16.data(), incY);

  if (M == 0 || N == 0) {
    for (std::size_t i = 0; i < y_size; ++i) {
      EXPECT_EQ(static_cast<float>(Y_fp16[i]), static_cast<float>(Y_before[i]))
        << "zero-dimension GEMV touched Y at i=" << i << " M=" << M
        << " N=" << N << " TransA=" << TransA;
    }
    return;
  }

  for (unsigned int k = 0; k < lenY; ++k) {
    const std::size_t y_idx = static_cast<std::size_t>(k) * incY;
    const float got = static_cast<float>(Y_fp16[y_idx]);
    const float ref = Y_fp32_ref[y_idx];
    const float abs_diff = std::abs(got - ref);
    const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
    EXPECT_TRUE(abs_diff <= 1e-2F || rel_diff <= 1e-2F)
      << "mismatch at k=" << k << " M=" << M << " N=" << N
      << " TransA=" << TransA << " alpha=" << alpha << " beta=" << beta
      << " lda=" << lda << " incX=" << incX << " incY=" << incY
      << " got=" << got << " ref=" << ref << " abs_diff=" << abs_diff
      << " rel_diff=" << rel_diff;
  }

  // Y gap slots (only present when incY > 1) must be untouched.
  if (incY > 1) {
    for (unsigned int k = 0; k < lenY; ++k) {
      const std::size_t base = static_cast<std::size_t>(k) * incY;
      for (unsigned int g = 1; g < incY; ++g) {
        const std::size_t gap_idx = base + g;
        if (gap_idx >= y_size) {
          break;
        }
        EXPECT_EQ(static_cast<float>(Y_fp16[gap_idx]),
                  static_cast<float>(Y_before[gap_idx]))
          << "Y gap was modified at idx=" << gap_idx << " k=" << k << " g=" << g
          << " incY=" << incY;
      }
    }
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_aligned_8x64) {
  // N multiple of 16: 16-element main loop only, no tails.
  run_sgemv_fp16_hgemv_test(8, 64);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_aligned_16x128) {
  run_sgemv_fp16_hgemv_test(16, 128);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_n8_tail_5x24) {
  // N=24 = 16 + 8: exercises the 8-element tail after the 16-element loop.
  run_sgemv_fp16_hgemv_test(5, 24);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_scalar_tail_7x31) {
  // N=31 = 16 + 8 + 7: exercises 16-loop, 8-tail, then scalar tail of 7.
  run_sgemv_fp16_hgemv_test(7, 31);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemv_fp16_noTrans_scalar_tail_only_5x9) {
  // N=9: skips 16-loop and 8-tail, scalar tail of 9.
  run_sgemv_fp16_hgemv_test(5, 9);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_aligned_16x64) {
  run_sgemv_fp16_hgemv_test(16, 64, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_unaligned_13x33) {
  // M=13 (AXPY row sweep tail), N=33 (8-element tail + scalar tail in
  // the FP32-scratch axpy update).
  run_sgemv_fp16_hgemv_test(13, 33, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_noTrans_alpha_beta_13x33) {
  run_sgemv_fp16_hgemv_test(13, 33, false, -0.75F, 0.25F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_transA_alpha_beta_13x33) {
  run_sgemv_fp16_hgemv_test(13, 33, true, 0.5F, -0.125F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_lda_noTrans_7x17) {
  // lda = N + 7: row-padded A.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 7);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_lda_transA_7x17) {
  run_sgemv_fp16_hgemv_test(7, 17, true, 1.0F, 0.0F, 7);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_incX_noTrans_7x17) {
  // incX=2 forces the non-contiguous X path.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 0, 2, 1);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_strided_incY_noTrans_7x17) {
  // incY=3 exercises Y-gap preservation in the writeback path.
  run_sgemv_fp16_hgemv_test(7, 17, false, 1.0F, 0.0F, 0, 1, 3);
}

TEST(nntrainer_cpu_backend_standalone,
     sgemv_fp16_strided_lda_incX_incY_transA_7x17) {
  // All three strides at once on the TransA path (non-contiguous Y32 init,
  // non-contiguous incX read, non-contiguous incY writeback).
  run_sgemv_fp16_hgemv_test(7, 17, true, 0.5F, -0.25F, 3, 2, 4);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_alpha_beta_boundary_cases) {
  /// @brief parameter set for one sub-case of this test
  struct Case {
    float alpha;
    float beta;
  };

  const std::vector<Case> cases = {
    {0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {2.5F, -1.0F}};
  for (const auto &tc : cases) {
    SCOPED_TRACE("alpha=" + std::to_string(tc.alpha) +
                 " beta=" + std::to_string(tc.beta));
    run_sgemv_fp16_hgemv_test(13, 33, false, tc.alpha, tc.beta);
    run_sgemv_fp16_hgemv_test(13, 33, true, tc.alpha, tc.beta);
  }
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_zero_dimension_cases) {
  run_sgemv_fp16_hgemv_test(0, 17, false, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(7, 0, false, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(0, 17, true, 1.0F, 0.0F);
  run_sgemv_fp16_hgemv_test(7, 0, true, 1.0F, 0.0F);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_single_row_or_col) {
  run_sgemv_fp16_hgemv_test(1, 33, false);
  run_sgemv_fp16_hgemv_test(33, 1, false);
  run_sgemv_fp16_hgemv_test(1, 33, true);
  run_sgemv_fp16_hgemv_test(33, 1, true);
}

TEST(nntrainer_cpu_backend_standalone, sgemv_fp16_beta_zero_does_not_read_Y) {
  // BLAS rule: when beta == 0, Y must not be read. Seed Y with NaN; if the
  // kernel reads it, NaN propagates into the result.
  nntrainer::init_backend();

  const _FP16 nan_v = static_cast<_FP16>(std::nan(""));

  auto check_path = [&](bool TransA) {
    const unsigned int M = 7;
    const unsigned int N = 17;
    const unsigned int lenY = TransA ? N : M;

    auto A = generate_random_vector<_FP16>(M * N, -0.25F, 0.25F);
    auto X = generate_random_vector<_FP16>(TransA ? M : N, -0.25F, 0.25F);
    std::vector<_FP16> Y(lenY, nan_v);

    nntrainer::sgemv(0, TransA, M, N, 1.0F, A.data(), N, X.data(), 1, 0.0F,
                     Y.data(), 1);

    for (unsigned int k = 0; k < lenY; ++k) {
      const float yv = static_cast<float>(Y[k]);
      EXPECT_FALSE(std::isnan(yv)) << "beta=0 path read NaN-seeded Y at k=" << k
                                   << " TransA=" << TransA << " value=" << yv;
    }
  };

  check_path(false);
  check_path(true);
}

/// Mixed-precision GEMV path (shgemv: A=FP32,X=FP16 / hsgemv: A=FP16,X=FP32),
/// FP32 output. Reference accumulates in double; see run_mixed_sgemm_test.
template <typename MatT, typename VecT>
static void run_mixed_sgemv_test(unsigned int M, unsigned int N,
                                 bool TransA = false, float alpha = 1.0F,
                                 float beta = 0.0F, unsigned int lda_extra = 0,
                                 unsigned int incX = 1, unsigned int incY = 1) {
  static_assert(std::is_same_v<MatT, float> != std::is_same_v<VecT, float>,
                "exactly one operand must be FP32 for mixed-precision GEMV");
  nntrainer::init_backend();

  const unsigned int lda = std::max(1u, N + lda_extra);
  const unsigned int lenX = TransA ? M : N;
  const unsigned int lenY = TransA ? N : M;

  const std::size_t a_size =
    std::max<std::size_t>(1, static_cast<std::size_t>(M) * lda);
  const std::size_t x_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenX) * std::max(incX, 1u));
  const std::size_t y_size = std::max<std::size_t>(
    1, static_cast<std::size_t>(lenY) * std::max(incY, 1u));

  auto A = generate_random_vector<MatT>(a_size, -0.25F, 0.25F);
  auto X = generate_random_vector<VecT>(x_size, -0.25F, 0.25F);
  auto Y = generate_random_vector<float>(y_size, -0.25F, 0.25F);
  auto Y_before = Y;

  std::vector<float> Y_ref(y_size);
  for (std::size_t i = 0; i < y_size; ++i) {
    Y_ref[i] = Y_before[i];
  }

  if (!TransA) {
    for (unsigned int i = 0; i < M; ++i) {
      double acc = 0.0;
      for (unsigned int j = 0; j < N; ++j) {
        acc += static_cast<double>(static_cast<float>(A[i * lda + j])) *
               static_cast<double>(static_cast<float>(X[j * incX]));
      }
      const std::size_t y_idx = static_cast<std::size_t>(i) * incY;
      Y_ref[y_idx] =
        static_cast<float>(static_cast<double>(alpha) * acc +
                           static_cast<double>(beta) * Y_ref[y_idx]);
    }
  } else {
    for (unsigned int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (unsigned int i = 0; i < M; ++i) {
        acc += static_cast<double>(static_cast<float>(A[i * lda + j])) *
               static_cast<double>(static_cast<float>(X[i * incX]));
      }
      const std::size_t y_idx = static_cast<std::size_t>(j) * incY;
      Y_ref[y_idx] =
        static_cast<float>(static_cast<double>(alpha) * acc +
                           static_cast<double>(beta) * Y_ref[y_idx]);
    }
  }

  if constexpr (std::is_same_v<MatT, float>) {
    nntrainer::shgemv(0, TransA, M, N, alpha, A.data(), lda, X.data(), incX,
                      beta, Y.data(), incY);
  } else {
    nntrainer::hsgemv(0, TransA, M, N, alpha, A.data(), lda, X.data(), incX,
                      beta, Y.data(), incY);
  }

  for (unsigned int k = 0; k < lenY; ++k) {
    const std::size_t y_idx = static_cast<std::size_t>(k) * incY;
    const float got = Y[y_idx];
    const float ref = Y_ref[y_idx];
    const float abs_diff = std::abs(got - ref);
    const float rel_diff = abs_diff / std::max(1.0F, std::abs(ref));
    EXPECT_TRUE(abs_diff <= 1e-3F || rel_diff <= 1e-3F)
      << "mismatch at k=" << k << " M=" << M << " N=" << N
      << " TransA=" << TransA << " alpha=" << alpha << " beta=" << beta
      << " incX=" << incX << " incY=" << incY << " got=" << got
      << " ref=" << ref << " abs_diff=" << abs_diff << " rel_diff=" << rel_diff;
  }
}

TEST(nntrainer_cpu_backend_standalone, shgemv_fp32xfp16_noTrans_16x128) {
  run_mixed_sgemv_test<float, _FP16>(16, 128);
}
TEST(nntrainer_cpu_backend_standalone,
     shgemv_fp32xfp16_transA_unaligned_13x33) {
  run_mixed_sgemv_test<float, _FP16>(13, 33, true);
}
TEST(nntrainer_cpu_backend_standalone, shgemv_fp32xfp16_alpha_beta_13x33) {
  run_mixed_sgemv_test<float, _FP16>(13, 33, false, 0.5F, 0.25F);
  run_mixed_sgemv_test<float, _FP16>(13, 33, true, 0.5F, 0.25F);
}
TEST(nntrainer_cpu_backend_standalone, hsgemv_fp16xfp32_noTrans_16x128) {
  run_mixed_sgemv_test<_FP16, float>(16, 128);
}
TEST(nntrainer_cpu_backend_standalone,
     hsgemv_fp16xfp32_transA_unaligned_13x33) {
  run_mixed_sgemv_test<_FP16, float>(13, 33, true);
}
TEST(nntrainer_cpu_backend_standalone,
     hsgemv_fp16xfp32_strided_incX_incY_7x17) {
  run_mixed_sgemv_test<_FP16, float>(7, 17, false, 1.0F, 0.0F, 3, 2, 2);
  run_mixed_sgemv_test<float, _FP16>(7, 17, true, 0.75F, 0.5F, 0, 2, 2);
}

#endif // defined(__x86_64__) || defined(_M_X64)

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
