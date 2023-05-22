/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4gemm.cpp

Abstract:

    This module implements the fp32 matrix multiplication with compressed
    weight tensor (right hand side). The assumption is the right hand side
    tensor can be pre-packed and compressed using int-4 quantization to save
    memory. Quantized weights are expanded to fp32 before matrix
    multiplication.

--*/

#include "q4common.h"

template<typename Q4TYPE, typename KERNEL>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    float* C,
    size_t ldc,
    const float* Bias,
    const float* A,
    size_t lda,
    const uint8_t* PackedB);

struct MLAS_FP_Q4_GEMM_KERNEL_DEFAULT {
    static constexpr size_t StrideM = 4;
};

template<>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK1,MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    float* C,
    size_t ldc,
    const float* Bias,
    const float* A,
    size_t lda,
    const uint8_t* PackedB)
{
    for (size_t m = 0; m < CountM; m++) {
        const uint8_t* b = PackedB;
        for (size_t n = 0; n < CountN; n += MLAS_Q4_N_STRIDE) {
            size_t cn = std::min(CountN - n, MLAS_Q4_N_STRIDE);

            float* sum = C + n;
            for (size_t i = 0; i < cn; i++) {
                sum[i] = Bias == nullptr ? 0.0f : Bias[n + i];
            }

            for (size_t k = 0; k < CountK; k += MLAS_Q4TYPE_BLK1::BlkLen) {
                size_t ck = std::min(CountK - k, MLAS_Q4TYPE_BLK1::BlkLen);

                for (size_t nn = 0; nn < cn; nn++) {
                    const float s = MlasQ4BlkScale<MLAS_Q4TYPE_BLK1>(b);
                    const uint8_t z = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b);
                    const uint8_t* pp = MlasQ4BlkData<MLAS_Q4TYPE_BLK1>(b);

                    for (size_t l = 0; l < ck; l += 2) {
                        const uint8_t vi = pp[l / 2];

                        const int8_t vi0 = vi & 0x0F;
                        const int8_t vi1 = vi >> 4;

                        const float v0 = (vi0 - z) * s;
                        const float v1 = (vi1 - z) * s;

                        sum[nn] += v0 * A[k + l];
                        if (l + 1 < ck) {
                            sum[nn] += v1 * A[k + l + 1];
                        }
                    }
                    b += MLAS_Q4TYPE_BLK1::BlobSize;
                }
                b += (MLAS_Q4_N_STRIDE - cn) * MLAS_Q4TYPE_BLK1::BlobSize;
            }
        }
        C += ldc;
        A += lda;
    }
    return CountM;
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    float* C,
    size_t ldc,
    const float* Bias,
    const float* A,
    size_t lda,
    const uint8_t* PackedB)
{
    for (size_t m = 0; m < CountM; m++) {
        const uint8_t* b = PackedB;
        for (size_t n = 0; n < CountN; n += MLAS_Q4_N_STRIDE) {
            size_t cn = std::min(CountN - n, MLAS_Q4_N_STRIDE);

            float* sum = C + n;
            for (size_t i = 0; i < cn; i++) {
                sum[i] = Bias == nullptr ? 0.0f : Bias[n + i];
            }

            for (size_t k = 0; k < CountK; k += MLAS_Q4TYPE_BLK0::BlkLen) {
                size_t ck = std::min(CountK - k, MLAS_Q4TYPE_BLK0::BlkLen);

                for (size_t nn = 0; nn < cn; nn++) {
                    const float s = MlasQ4BlkScale<MLAS_Q4TYPE_BLK0>(b);
                    const uint8_t* pp = MlasQ4BlkData<MLAS_Q4TYPE_BLK0>(b);

                    for (size_t l = 0; l < ck; l += 2) {
                        const uint8_t vi = pp[l / 2];

                        const int vi0 = (vi & 0x0F) - 8;
                        const int vi1 = (vi >> 4) - 8;

                        const float v0 = vi0 * s;
                        const float v1 = vi1 * s;

                        sum[nn] += v0 * A[k + l];
                        if (l + 1 < ck) {
                            sum[nn] += v1 * A[k + l + 1];
                        }
                    }
                    b += MLAS_Q4TYPE_BLK0::BlobSize;
                }
                b += (MLAS_Q4_N_STRIDE - cn) * MLAS_Q4TYPE_BLK0::BlobSize;
            }
        }
        C += ldc;
        A += lda;
    }
    return CountM;
}


template <typename Q4TYPE, typename KERNEL>
void MLASCALL
MlasQ4GemmOperation(
    const size_t K,
    const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;

    const size_t k_blks = MlasDivRoundup(K, MLAS_QUANT4_BLK_LEN);
    const float* A = DataParams->A + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)DataParams->B;
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;
    const float* Bias = DataParams->Bias;

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        CountN = std::min(RangeCountN - n, (size_t)128);

        //
        // Step through each slice of matrix A along the M dimension.
        //
        const float* bias = (Bias == nullptr) ? nullptr : Bias + RangeStartN + n;
        const uint8_t* b_col = PackedB + (RangeStartN + n) * k_blks * (typename Q4TYPE::BlobSize);
        float* c_blk = C + n;
        const float* a_row = A;

        size_t RowsRemaining = RangeCountM;
        while (RowsRemaining > 0) {
            auto RowsHandled =
                MlasQ4GemmKernel<Q4TYPE, KERNEL>(RowsRemaining, CountN, K, c_blk, ldc, bias, a_row, lda, b_col);

            if (DataParams->OutputProcessor != nullptr) {
                DataParams->OutputProcessor->Process(
                    DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                    RowsHandled, CountN, DataParams->ldc);
            }

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}

void
MLASCALL
MlasQ4GemmBatch(
    MLAS_BLK_QUANT_TYPE QType,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool
    )
{
    //const MLAS_Q4GEMM_DISPATCH* dispatch = MlasQ4GemmGetDispatch();
    //MLAS_Q4GEMM_OPERATION* operation = dispatch->Operation;
    MLAS_UNREFERENCED_PARAMETER(QType);
    auto *operation =
        QType == BlkQ4Sym ? 
        MlasQ4GemmOperation<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>
        : MlasQ4GemmOperation<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>;

    if (ThreadPool == nullptr) {
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            auto Data = &DataParams[gemm_i];
            operation(K, Data, 0, M, 0, N);
        }
        return;
    }

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    const size_t StrideM = 4;  // dispatch->StrideM;

    size_t nc = N;
    if (ThreadsPerGemm > 1) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(nc, MlasDivRoundup(nc, max_nc * MLAS_QGEMM_STRIDEN_THREAD_ALIGN) *
                                  MLAS_QGEMM_STRIDEN_THREAD_ALIGN);
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        auto Data = &DataParams[gemm_i];

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        operation(K, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}
