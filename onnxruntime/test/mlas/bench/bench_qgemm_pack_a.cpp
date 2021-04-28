// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"
#include "core/util/qmath.h"

#include <stdexcept>
#include <memory>
#include <numeric>
#include <algorithm>

//
// Assuming B is always pre-packed.
// Comparing ( QGemm) vs (prepack A + QGemm)
//

static void ComputeGemm(
    size_t M,
    size_t K,
    size_t N,
    const uint8_t* a_data, 
    uint8_t a_zero_point,
    void* packed_b,
    uint8_t b_zero_point,
    bool b_is_signed,
    int32_t* y_data,
    onnxruntime::concurrency::ThreadPool* tp)
{

  MLAS_GEMM_U8X8_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = M;
  gemm_shape.N = N;
  gemm_shape.K = K;
  gemm_shape.BIsSigned = b_is_signed;

  MLAS_GEMM_U8X8_DATA_PARAMS gemm_data;

  gemm_data.A = a_data;
  gemm_data.lda = gemm_shape.K;
  gemm_data.ZeroPointA = a_zero_point;
  gemm_data.BIsPacked = true;
  gemm_data.B = packed_b;
  gemm_data.ldb = gemm_shape.N;
  gemm_data.ZeroPointB = &b_zero_point;
  gemm_data.C = y_data;
  gemm_data.ldc = gemm_shape.N;

  MlasGemm(gemm_shape, gemm_data, tp);
}

static void ComputePackAGemm(
    size_t M,
    size_t K,
    size_t N,
    const uint8_t* a_data, 
    uint8_t a_zero_point,
    uint8_t* a_data_packed,
    void* packed_b,
    uint8_t b_zero_point,
    bool b_is_signed,
    int32_t* y_data,
    onnxruntime::concurrency::ThreadPool* tp) {
  MlasGemmPackA(M, K, b_is_signed, a_data, K, a_data_packed, tp);

  // batch gemm
  MLAS_GEMM_U8X8_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = M;
  gemm_shape.N = N;
  gemm_shape.K = K;
  gemm_shape.BIsSigned = b_is_signed;

  MLAS_GEMM_U8X8_DATA_PARAMS gemm_data;

  gemm_data.A = a_data_packed;
  gemm_data.AIsPacked = true;
  gemm_data.lda = gemm_shape.K;
  gemm_data.ZeroPointA = a_zero_point;
  gemm_data.BIsPacked = true;
  gemm_data.B = packed_b;
  gemm_data.ldb = gemm_shape.N;
  gemm_data.ZeroPointB = &b_zero_point;
  gemm_data.C = y_data;
  gemm_data.ldc = gemm_shape.N;

  MlasGemm(gemm_shape, gemm_data, tp);
}

static const std::vector<std::string> qgemm_pack_arg_names = {"M", "N", "K", "Batch", "Threads"};

void PackGemmA(benchmark::State& state, bool b_is_signed, bool quant_pack_a) {

  const uint8_t a_zero_point = 15;
  const uint8_t b_zero_point = 5;

  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Batch must greater than 0!");
  if (state.range(4) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  const size_t batch = static_cast<size_t>(state.range(3));
  const size_t threads = static_cast<size_t>(state.range(4));

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A_holder = RandomVectorUniform<uint8_t>(static_cast<size_t>(M * K * batch) + MLAS_BENCH_BUF_ALIGN, uint8_t(5), uint8_t(245));

  auto B_holder = RandomVectorUniform<uint8_t>(static_cast<size_t>(N * K * batch) + MLAS_BENCH_BUF_ALIGN, uint8_t(5), uint8_t(245));
  std::vector<int32_t> C_holder(static_cast<size_t>(M * N * batch) + MLAS_BENCH_BUF_ALIGN);
  std::vector<uint8_t> pack_b_holder;
  std::vector<uint8_t> pack_a_holder;

  size_t packed_b_size = MlasGemmPackBSize(N, K, b_is_signed);
  pack_b_holder.resize(packed_b_size * batch + MLAS_BENCH_BUF_ALIGN);
  MlasGemmPackB(N, K, AlignAddr(B_holder.data()), N, b_is_signed, AlignAddr(pack_b_holder.data()));

  size_t packed_a_size = MlasGemmPackASize(M, K, b_is_signed);
  pack_a_holder.resize(packed_a_size * batch + MLAS_BENCH_BUF_ALIGN);

  for (auto _ : state) {
    if (quant_pack_a) {
      ComputePackAGemm(M, K, N, AlignAddr(A_holder.data()), a_zero_point,
                       AlignAddr(pack_a_holder.data()),
                       AlignAddr(pack_b_holder.data()), b_zero_point, b_is_signed,
                       AlignAddr(C_holder.data()), tp.get());
    } else {
      ComputeGemm(M, K, N, AlignAddr(A_holder.data()), a_zero_point, AlignAddr(pack_b_holder.data()),
                  b_zero_point, b_is_signed, AlignAddr(C_holder.data()), tp.get());
    }
  }
}



static void QGemmSize(benchmark::internal::Benchmark* b) {
  b->ArgNames(qgemm_pack_arg_names);
  b->Args({512, 3072, 768, 1, 1});
  b->Args({512, 3072, 768, 1, 4});
  b->Args({512, 3072, 768, 1, 8});
  b->Args({512, 3072, 768, 1, 16});
  b->Args({512, 768, 3072, 1, 1});
  b->Args({512, 768, 3072, 1, 4});
  b->Args({512, 768, 3072, 1, 8});
  b->Args({512, 768, 3072, 1, 16});
  b->Args({512, 768, 768, 1, 1});
  b->Args({512, 768, 768, 1, 4});
  b->Args({512, 768, 768, 1, 8});
  b->Args({512, 768, 768, 1, 16});
  b->Args({512, 64, 512, 1, 1});
  b->Args({512, 64, 512, 1, 4});
  b->Args({512, 64, 512, 1, 8});
  b->Args({512, 64, 512, 1, 16});
}

BENCHMARK_CAPTURE(PackGemmA, SignedB________, true, false)->Apply(QGemmSize)->UseRealTime();
BENCHMARK_CAPTURE(PackGemmA, SignedB___PackA, true, true)->Apply(QGemmSize)->UseRealTime();
BENCHMARK_CAPTURE(PackGemmA, UnSignedB______, false, false)->Apply(QGemmSize)->UseRealTime();
BENCHMARK_CAPTURE(PackGemmA, UnSignedB_PackA, false, true)->Apply(QGemmSize)->UseRealTime();
