// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <stdexcept>
#include <memory>
#include <numeric>
#include <algorithm>


void QUANTPACKA(benchmark::State& state, bool b_is_signed) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("K must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t K = static_cast<size_t>(state.range(1));

  auto A_holder = RandomVectorUniform<float>(static_cast<size_t>(M * K), float(-100), float(100));

  size_t packed_a_size = MlasGemmPackASize(M, K, b_is_signed);
  std::vector<uint8_t> packed_a_holder(packed_a_size);

  for (auto _ : state) {
    MlasQuantizeLinearPackA(M, K, A_holder.data(), K, b_is_signed, 7.89f, uint8_t(17), packed_a_holder.data(), nullptr);
  }
}

void QUANTPACKB(benchmark::State& state, bool b_is_signed) {
  if (state.range(0) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("K must greater than 0!");

  const size_t N = static_cast<size_t>(state.range(0));
  const size_t K = static_cast<size_t>(state.range(1));

  auto A_holder = RandomVectorUniform<float>(static_cast<size_t>(N * K), float(-100), float(100));

  size_t packed_a_size = MlasGemmPackBSize(N, K, b_is_signed);
  std::vector<uint8_t> packed_a_holder(packed_a_size);

  for (auto _ : state) {
    MlasQuantizeLinearPackB(N, K, A_holder.data(), N, b_is_signed, 7.89f, uint8_t(17), packed_a_holder.data(), nullptr);
  }
}


void QUANT(benchmark::State& state) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t K = static_cast<size_t>(state.range(1));

  auto A_holder = RandomVectorUniform<float>(static_cast<size_t>(M * K), float(-100), float(100));
  std::vector<uint8_t> result_holder(M * K);

  for (auto _ : state) {
    MlasQuantizeLinear(A_holder.data(), result_holder.data(), M * K, 7.89f, uint8_t(16));
  }
}

static void QSizes(benchmark::internal::Benchmark* b) {
  b->Args({5, 1000000});
  b->Args({25, 200000});
  b->Args({500, 10000});
  b->Args({1000, 5000});
  b->Args({5000, 1000});
  b->Args({10000, 500});
  b->Args({50000, 100});
  b->Args({100000, 50});
  b->Args({200000, 25});
  b->Args({1000000, 5});
}

BENCHMARK_CAPTURE(QUANT, T,)->Apply(QSizes)->UseRealTime();
BENCHMARK_CAPTURE(QUANTPACKA, SignedB, true)->Apply(QSizes)->UseRealTime();
BENCHMARK_CAPTURE(QUANTPACKA, UnsignedB, false)->Apply(QSizes)->UseRealTime();
BENCHMARK_CAPTURE(QUANTPACKB, SignedB, true)->Apply(QSizes)->UseRealTime();
BENCHMARK_CAPTURE(QUANTPACKB, UnsignedB, false)->Apply(QSizes)->UseRealTime();
