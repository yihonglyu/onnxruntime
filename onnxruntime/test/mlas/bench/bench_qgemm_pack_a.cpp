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
// Comparing ( QGemm) vs (pack A + QGemm)
//

static const std::vector<std::string> qgemm_pack_arg_names = {"M", "N", "K", "Batch", "Threads"};

void PackGemmA(benchmark::State& state, bool b_is_signed, bool pack_a) {

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

  size_t packed_b_size = MlasGemmPackBSize(N, K, b_is_signed);
  std::vector<uint8_t> pack_b_holder(packed_b_size * batch + MLAS_BENCH_BUF_ALIGN);

  size_t packed_a_size = MlasGemmPackASize(M, K, b_is_signed);
  std::vector<uint8_t> pack_a_holder(packed_a_size * batch + MLAS_BENCH_BUF_ALIGN);

  MLAS_GEMM_U8X8_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(M);
  gemm_shape.N = static_cast<size_t>(N);
  gemm_shape.K = static_cast<size_t>(K);
  gemm_shape.BIsSigned = b_is_signed;

  std::vector<MLAS_GEMM_U8X8_DATA_PARAMS> gemm_data_vec(batch);
  for (size_t i = 0; i < batch; i++) {
    auto& gemm_params = gemm_data_vec[i];
    gemm_params.lda = gemm_shape.K;
    gemm_params.ZeroPointA = a_zero_point;
    gemm_params.ZeroPointB = &b_zero_point;
    gemm_params.ldc = gemm_shape.N;
    gemm_params.A = AlignAddr(A_holder.data()) + M * K * i;
    gemm_params.PackedA = (void*)(AlignAddr(pack_a_holder.data()) + packed_a_size * i);

    uint8_t* BBuf = AlignAddr(B_holder.data()) + N * K * i;
    void* PackedBBuf = (void*)(AlignAddr(pack_b_holder.data()) + packed_b_size * i);
    MlasGemmPackB(N, K, BBuf, N, b_is_signed, PackedBBuf);
    gemm_params.BIsPacked = true;
    gemm_params.PackedB = PackedBBuf;
    gemm_params.B = nullptr;

    gemm_params.ldb = gemm_shape.N;
    gemm_params.C = AlignAddr(C_holder.data()) + M * N * i;
  }


  for (auto _ : state) {
    if (pack_a) {
      for (size_t i = 0; i < batch; i++) {
        auto& gemm_params = gemm_data_vec[i];
        MlasGemmPackA(M, K, b_is_signed, AlignAddr(A_holder.data()) + M * K * i, gemm_params.lda, gemm_params.PackedA, tp.get());
        gemm_params.A = nullptr;
      }
    }
    MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch, tp.get());
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
