// SPDX-License-Identifier: Apache-2.0
/**
 * @file  benchmark_x86_backend.cpp
 * @date  24 March 2026
 * @brief Performance benchmarks for cpu compute backend
 * @see   https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho
 * @bug   No known bugs except for NYI items
 */

#include "benchmark_utils.h"
#include "nntrainer_test_util.h"

#include <cpu_backend.h>
#include <fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Global benchmark configuration (settable via command-line)
//
//   --bench_iters=N     Number of measured iterations per test (default: 1000)
//   --bench_warmup=N    Number of warmup iterations (default: 10)
//   --bench_sizes=S     Comma-separated element sizes (default: 256,1024,4096)
// ============================================================================

static unsigned int g_bench_iters = 1000;
static unsigned int g_bench_warmup = 10;
static std::vector<unsigned int> g_bench_sizes = {256, 1024, 4096};

// ============================================================================
// Helpers
// ============================================================================

/**
 * @brief Parse an unsigned integer benchmark option.
 *
 * @param text input option value
 * @param value parsed output value
 * @param allow_zero true if zero is accepted
 * @return true if parsing succeeded
 */
static bool parse_uint_arg(const std::string &text, unsigned int &value,
                           bool allow_zero) {
  if (text.empty()) {
    return false;
  }

  unsigned long long parsed = 0;
  char extra = '\0';
  std::istringstream ss(text);
  if (!(ss >> parsed) || (ss >> extra)) {
    return false;
  }
  if ((!allow_zero && parsed == 0) ||
      parsed > std::numeric_limits<unsigned int>::max()) {
    return false;
  }

  value = static_cast<unsigned int>(parsed);
  return true;
}

/**
 * @brief Format a numeric GoogleTest property value without integer narrowing.
 *
 * @param value numeric value to record
 * @return string-formatted property value
 */
static std::string make_record_property_value(double value) {
  return std::to_string(value);
}

/**
 * @brief Convert FP32 values to raw FP16 bit-pattern values.
 *
 * @param f32_vec input FP32 vector
 * @return converted FP16 bit-pattern vector
 */
static inline std::vector<uint16_t>
convert_f32_to_f16_u16(const std::vector<float> &f32_vec) {
  std::vector<uint16_t> vec(f32_vec.size());
  for (size_t i = 0; i < f32_vec.size(); i++) {
    vec[i] = nntrainer::compute_fp32_to_fp16(f32_vec[i]);
  }
  return vec;
}

// ============================================================================
// Element-wise Benchmarks (FP32 vs FP16 interleaved)
// ============================================================================

/**
 * @brief Test fixture for element-wise operation benchmarks
 */
class Bench_EleOps : public ::testing::Test {};

TEST_F(Bench_EleOps, ele_sub) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X = generate_random_vector<float>(N);
      auto Y = generate_random_vector<float>(N, 0.1f, 1.0f);
      std::vector<float> Z(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::ele_sub(N, X.data(), Y.data(), Z.data(), 1.f, 0.f, 1, 1);
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      m.total_bytes = 3 * N * sizeof(float);
      bench::report("ele_sub", "FP32", "N=" + std::to_string(N), stats, m);
    }

#ifdef ENABLE_FP16
    {
      auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
      auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
      std::vector<uint16_t> Z(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::ele_sub(N, (const _FP16 *)X_u16.data(),
                             (const _FP16 *)Y_u16.data(), (_FP16 *)Z.data(),
                             1.f, 0.f, 1, 1);
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      m.total_bytes = 3 * N * sizeof(uint16_t);
      bench::report("ele_sub", "FP16", "N=" + std::to_string(N), stats, m);
    }
#endif
  }
}

TEST_F(Bench_EleOps, ele_div) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X = generate_random_vector<float>(N);
      auto Y = generate_random_vector<float>(N, 0.1f, 2.0f);
      std::vector<float> Z(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::ele_div(N, X.data(), Y.data(), Z.data(), 1.f, 0.f, 1, 1);
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      m.total_bytes = 3 * N * sizeof(float);
      bench::report("ele_div", "FP32", "N=" + std::to_string(N), stats, m);
    }

#ifdef ENABLE_FP16
    {
      auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
      auto Y_u16 =
        convert_f32_to_f16_u16(generate_random_vector<float>(N, 0.1f, 2.0f));
      std::vector<uint16_t> Z(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::ele_div(N, (const _FP16 *)X_u16.data(),
                             (const _FP16 *)Y_u16.data(), (_FP16 *)Z.data(),
                             1.f, 0.f, 1, 1);
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      m.total_bytes = 3 * N * sizeof(uint16_t);
      bench::report("ele_div", "FP16", "N=" + std::to_string(N), stats, m);
    }
#endif
  }
}

#ifdef ENABLE_FP16
TEST_F(Bench_EleOps, ele_mul) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    std::vector<uint16_t> Z(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::ele_mul(N, (const _FP16 *)X_u16.data(),
                           (const _FP16 *)Y_u16.data(), (_FP16 *)Z.data(), 1.f,
                           0.f, 1, 1);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 3 * N * sizeof(uint16_t);
    bench::report("ele_mul", "FP16", "N=" + std::to_string(N), stats, m);
  }
}

TEST_F(Bench_EleOps, ele_add) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    std::vector<uint16_t> Z(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::ele_add(N, (const _FP16 *)X_u16.data(),
                           (const _FP16 *)Y_u16.data(), (_FP16 *)Z.data(), 1.f,
                           0.f, 1, 1);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 3 * N * sizeof(uint16_t);
    bench::report("ele_add", "FP16", "N=" + std::to_string(N), stats, m);
  }
}
#endif

// ============================================================================
// Activation / Reduction Benchmarks (FP32 vs FP16 interleaved)
// ============================================================================

/**
 * @brief Test fixture for activation and reduction benchmarks
 */
class Bench_Activations : public ::testing::Test {};

TEST_F(Bench_Activations, softmax) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X = generate_random_vector<float>(N);
      std::vector<float> Y(N);

      auto stats =
        bench::measure([&]() { nntrainer::softmax(N, X.data(), Y.data()); },
                       g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("softmax", "FP32", "N=" + std::to_string(N), stats, m);
    }

#ifdef ENABLE_FP16
    {
      auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
      std::vector<uint16_t> Y(N);

      auto stats = bench::measure(
        [&]() {
          nntrainer::softmax(N, (_FP16 *)X_u16.data(), (_FP16 *)Y.data());
        },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("softmax", "FP16", "N=" + std::to_string(N), stats, m);
    }
#endif
  }
}

TEST_F(Bench_Activations, tanh_gelu) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto X = generate_random_vector<float>(N);
    std::vector<float> Y(N);

    auto stats =
      bench::measure([&]() { nntrainer::tanh_gelu(N, X.data(), Y.data()); },
                     g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    bench::report("tanh_gelu", "FP32", "N=" + std::to_string(N), stats, m);
  }
}

TEST_F(Bench_Activations, tanh_gelu_mul) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto Y_in = generate_random_vector<float>(N);
    auto Z_in = generate_random_vector<float>(N);
    std::vector<float> X(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::tanh_gelu_mul(N, X.data(), Y_in.data(), Z_in.data());
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    bench::report("tanh_gelu_mul", "FP32", "N=" + std::to_string(N), stats, m);
  }
}

TEST_F(Bench_Activations, inv_sqrt_inplace) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X_orig = generate_random_vector<float>(N, 0.01f, 10.0f);
      auto X_tmp = X_orig;

      auto stats = bench::measure_with_setup(
        [&]() { std::copy(X_orig.begin(), X_orig.end(), X_tmp.begin()); },
        [&]() { nntrainer::inv_sqrt_inplace(N, X_tmp.data()); }, g_bench_warmup,
        g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("inv_sqrt_inplace", "FP32", "N=" + std::to_string(N), stats,
                    m);
    }

#ifdef ENABLE_FP16
    {
      auto X_orig =
        convert_f32_to_f16_u16(generate_random_vector<float>(N, 0.01f, 10.0f));
      auto X_u16 = X_orig;

      auto stats = bench::measure_with_setup(
        [&]() { X_u16 = X_orig; },
        [&]() { nntrainer::inv_sqrt_inplace(N, (_FP16 *)X_u16.data()); },
        g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("inv_sqrt_inplace", "FP16", "N=" + std::to_string(N), stats,
                    m);
    }
#endif
  }
}

TEST_F(Bench_Activations, max_val) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));

    {
      auto X = generate_random_vector<float>(N);

      float val = 0.0f;
      auto stats =
        bench::measure([&]() { val = nntrainer::max_val(N, X.data()); },
                       g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("max_val", "FP32", "N=" + std::to_string(N), stats, m);
    }

#ifdef ENABLE_FP16
    {
      auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));

      auto stats =
        bench::measure([&]() { nntrainer::max_val(N, (_FP16 *)X_u16.data()); },
                       g_bench_warmup, g_bench_iters);

      bench::Metrics m;
      m.num_elements = N;
      bench::report("max_val", "FP16", "N=" + std::to_string(N), stats, m);
    }
#endif
  }
}

#ifdef ENABLE_FP16
TEST_F(Bench_Activations, swiglu) {
  for (const unsigned int N : g_bench_sizes) {
    SCOPED_TRACE("N=" + std::to_string(N));
    auto Y_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    auto Z_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    std::vector<uint16_t> X(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::swiglu(N, (_FP16 *)X.data(), (_FP16 *)Y_u16.data(),
                          (_FP16 *)Z_u16.data());
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    bench::report("swiglu", "FP16", "N=" + std::to_string(N), stats, m);
  }
}
#endif

// ============================================================================
// RMS Norm Benchmarks (FP32 vs FP16 interleaved)
// ============================================================================

/**
 * @brief Test fixture for RMS normalization benchmarks
 */
class Bench_RmsNorm
  : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int>> {};

TEST_P(Bench_RmsNorm, rms_norm) {
  auto [H, W] = GetParam();
  const unsigned int H_v = H;
  const unsigned int W_v = W;
  const unsigned int N = H * W;
  float epsilon = 1e-6f;
  std::string sz = "H=" + std::to_string(H) + ",W=" + std::to_string(W);

  {
    auto X = generate_random_vector<float>(N);
    std::vector<float> Y(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(X.data(), Y.data(), H_v,
                                                     W_v, epsilon);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 2 * N * sizeof(float);
    bench::report("rms_norm", "FP32", sz, stats, m);
  }

#ifdef ENABLE_FP16
  {
    auto X_u16 = convert_f32_to_f16_u16(generate_random_vector<float>(N));
    std::vector<uint16_t> Y(N);

    auto stats = bench::measure(
      [&]() {
        nntrainer::rms_norm_wrt_width_fp16_intrinsic<_FP16>(
          (const _FP16 *)X_u16.data(), (_FP16 *)Y.data(), H_v, W_v, epsilon);
      },
      g_bench_warmup, g_bench_iters);

    bench::Metrics m;
    m.num_elements = N;
    m.total_bytes = 2 * N * sizeof(uint16_t);
    bench::report("rms_norm", "FP16", sz, stats, m);
  }
#endif
}

GTEST_PARAMETER_TEST(
  Dims, Bench_RmsNorm,
  ::testing::Values(std::make_tuple(1u, 128u), std::make_tuple(1u, 512u),
                    std::make_tuple(4u, 256u), std::make_tuple(4u, 1024u),
                    std::make_tuple(16u, 256u), std::make_tuple(16u, 1024u),
                    std::make_tuple(16u, 3072u)));

// ============================================================================
// Main
// ============================================================================

/**
 * @brief Parse benchmark-specific command-line arguments.
 *
 * @param argc argument count, updated after removing benchmark options
 * @param argv argument array, compacted in-place for GoogleTest
 */
static void parse_bench_args(int *argc, char **argv) {
  int out = 1;
  for (int i = 1; i < *argc; ++i) {
    if (strncmp(argv[i], "--bench_iters=", 14) == 0) {
      unsigned int v = 0;
      if (parse_uint_arg(argv[i] + 14, v, false)) {
        g_bench_iters = v;
      }
    } else if (strncmp(argv[i], "--bench_warmup=", 15) == 0) {
      unsigned int v = 0;
      if (parse_uint_arg(argv[i] + 15, v, true)) {
        g_bench_warmup = v;
      }
    } else if (strncmp(argv[i], "--bench_sizes=", 14) == 0) {
      std::vector<unsigned int> parsed_sizes;
      std::istringstream ss(argv[i] + 14);
      std::string token;
      while (std::getline(ss, token, ',')) {
        unsigned int v = 0;
        if (parse_uint_arg(token, v, false)) {
          parsed_sizes.push_back(v);
        }
      }
      if (!parsed_sizes.empty()) {
        g_bench_sizes = parsed_sizes;
      }
    } else {
      argv[out++] = argv[i];
    }
  }
  *argc = out;
}

/**
 * @brief Benchmark binary entry point.
 *
 * @param argc argument count
 * @param argv argument array
 * @return GoogleTest result code
 */
int main(int argc, char **argv) {
  parse_bench_args(&argc, argv);

  nntrainer::init_backend();
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "  Config: iters=" << g_bench_iters
            << ", warmup=" << g_bench_warmup << ", sizes={";
  for (size_t i = 0; i < g_bench_sizes.size(); ++i) {
    if (i > 0)
      std::cout << ",";
    std::cout << g_bench_sizes[i];
  }
  std::cout << "}" << std::endl;

  bench::print_separator("nntrainer CPU Backend Benchmark");
  bench::print_header();

  return RUN_ALL_TESTS();
}
