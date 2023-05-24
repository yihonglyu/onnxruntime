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

#include <type_traits>

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


/**
 * @brief Horizontally sum 4 vectors and store
 *        the results in the returned vector
 */
static
MLAS_FORCEINLINE
__m128
FoldAccumulators(
    const __m512& acc0,
    const __m512& acc1,
    const __m512& acc2,
    const __m512& acc3
    )
{
    __m512 acc_lo01 = _mm512_unpacklo_ps(acc0, acc1);
    __m512 acc_hi01 = _mm512_unpackhi_ps(acc0, acc1);
    __m512 acc_lo23 = _mm512_unpacklo_ps(acc2, acc3);
    __m512 acc_hi23 = _mm512_unpackhi_ps(acc2, acc3);

    __m512 acc_lo0123 = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23)));
    __m512 acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);

    __m256 acc_y =
        _mm256_add_ps(_mm512_extractf32x8_ps(acc_lo0123, 0), _mm512_extractf32x8_ps(acc_lo0123, 1));
    return _mm_add_ps(_mm256_extractf32x4_ps(acc_y, 0), _mm256_extractf32x4_ps(acc_y, 1));
}


template<typename Q4Type>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernelAvx512f(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    float* C,
    size_t ldc,
    const float* Bias,
    const float* A,
    size_t lda,
    const uint8_t* PackedB
    )
{
    const __m256i lowMask = _mm256_set1_epi8(0xF);

    for (size_t m = 0; m < CountM; m++) {
        const uint8_t* b = PackedB;

        auto* sum_ptr = C;
        auto* bias_ptr = Bias;
        int64_t nblk = (int64_t)(CountN) - MLAS_Q4_N_STRIDE;
        while (nblk >= 0) {
            static_assert(MLAS_Q4_N_STRIDE == 4);
            __m512 acc_lo0 = _mm512_setzero();
            __m512 acc_lo1 = _mm512_setzero();
            __m512 acc_lo2 = _mm512_setzero();
            __m512 acc_lo3 = _mm512_setzero();

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                size_t ck = std::min(CountK - k, (typename Q4Type::BlkLen));

                // Load A row vectors
                uint32_t mask = 0xffffffff >> (typename Q4Type::BlkLen - ck);
                __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k);

                mask = mask >> 16;
                __m512 av_hi = mask == 0 ? _mm512_setzero_ps()
                                         : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + 16);

                // Load 4 B column vectors (quantized to int4 blobs)
                const __m512 scale_v0 = _mm512_set1_ps(MlasQ4BlkScale<Q4Type>(b));
                uint8_t zp0 = 8;
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>)
                    zp0 = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b);
                const __m128i bvi4_0 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b));
                b += Q4Type::BlobSize;

                const __m512 scale_v1 = _mm512_set1_ps(MlasQ4BlkScale<Q4Type>(b));
                uint8_t zp1 = 8;
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>)
                    zp1 = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b);
                const __m128i bvi4_1 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b));
                b += Q4Type::BlobSize;

                const __m512 scale_v2 = _mm512_set1_ps(MlasQ4BlkScale<Q4Type>(b));
                uint8_t zp2 = 8;
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>)
                    zp2 = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b);
                const __m128i bvi4_2 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b));
                b += Q4Type::BlobSize;

                const __m512 scale_v3 = _mm512_set1_ps(MlasQ4BlkScale<Q4Type>(b));
                uint8_t zp3 = 8;
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>)
                    zp3 = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b);
                const __m128i bvi4_3 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b));
                b += Q4Type::BlobSize;

                // expand 4b into byte array
                __m256i bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_0, 4), bvi4_0);
                __m256i bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_1, 4), bvi4_1);
                __m256i bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_2, 4), bvi4_2);
                __m256i bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_3, 4), bvi4_3);
                bytes0 = _mm256_and_si256(lowMask, bytes0);
                bytes1 = _mm256_and_si256(lowMask, bytes1);
                bytes2 = _mm256_and_si256(lowMask, bytes2);
                bytes3 = _mm256_and_si256(lowMask, bytes3);

                // Subtract zero-point from the integers
                bytes0 = _mm256_sub_epi8(bytes0, _mm256_set1_epi8(zp0));
                bytes1 = _mm256_sub_epi8(bytes1, _mm256_set1_epi8(zp1));
                bytes2 = _mm256_sub_epi8(bytes2, _mm256_set1_epi8(zp2));
                bytes3 = _mm256_sub_epi8(bytes3, _mm256_set1_epi8(zp3));

                // Convert to 16-bit int
                const __m256i vx16_lo0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
                const __m256i vx16_hi0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
                const __m256i vx16_lo1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
                const __m256i vx16_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
                const __m256i vx16_lo2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
                const __m256i vx16_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
                const __m256i vx16_lo3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
                const __m256i vx16_hi3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

                // Convert to 32-bit int -> float 32
                __m512 bvf_lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
                __m512 bvf_hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
                __m512 bvf_lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
                __m512 bvf_hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
                __m512 bvf_lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
                __m512 bvf_hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
                __m512 bvf_lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
                __m512 bvf_hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));
                bvf_lo0 = _mm512_mul_ps(bvf_lo0, scale_v0);
                bvf_hi0 = _mm512_mul_ps(bvf_hi0, scale_v0);
                bvf_lo1 = _mm512_mul_ps(bvf_lo1, scale_v1);
                bvf_hi1 = _mm512_mul_ps(bvf_hi1, scale_v1);
                bvf_lo2 = _mm512_mul_ps(bvf_lo2, scale_v2);
                bvf_hi2 = _mm512_mul_ps(bvf_hi2, scale_v2);
                bvf_lo3 = _mm512_mul_ps(bvf_lo3, scale_v3);
                bvf_hi3 = _mm512_mul_ps(bvf_hi3, scale_v3);

                acc_lo0 = _mm512_fmadd_ps(bvf_lo0, av_lo, acc_lo0);
                acc_lo0 = _mm512_fmadd_ps(bvf_hi0, av_hi, acc_lo0);
                acc_lo1 = _mm512_fmadd_ps(bvf_lo1, av_lo, acc_lo1);
                acc_lo1 = _mm512_fmadd_ps(bvf_hi1, av_hi, acc_lo1);
                acc_lo2 = _mm512_fmadd_ps(bvf_lo2, av_lo, acc_lo2);
                acc_lo2 = _mm512_fmadd_ps(bvf_hi2, av_hi, acc_lo2);
                acc_lo3 = _mm512_fmadd_ps(bvf_lo3, av_lo, acc_lo3);
                acc_lo3 = _mm512_fmadd_ps(bvf_hi3, av_hi, acc_lo3);
            }

            __m128 acc_x = FoldAccumulators(acc_lo0, acc_lo1, acc_lo2, acc_lo3);

            if (Bias != nullptr) {
                acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
            }

            _mm_store_ps(sum_ptr, acc_x);

            sum_ptr += MLAS_Q4_N_STRIDE;
            bias_ptr += MLAS_Q4_N_STRIDE;
            nblk -= MLAS_Q4_N_STRIDE;
        }

        // left over columns less than 4 ?
        nblk += MLAS_Q4_N_STRIDE;
        if (nblk > 0) {
            __m512 acc_lo[MLAS_Q4_N_STRIDE]{};

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                size_t ck = std::min(CountK - k, (typename Q4Type::BlkLen));

                uint32_t mask = 0xffffffff >> ((typename Q4Type::BlkLen) - ck);
                __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k);

                mask = mask >> 16;
                __m512 av_hi = mask == 0 ? _mm512_setzero_ps()
                    : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + 16);

                for (int64_t nn = 0; nn < nblk; nn++) {
                    const __m512 scale_v = _mm512_set1_ps(MlasQ4BlkScale<Q4Type>(b));

                    const __m128i bvi4 =
                        _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b));
                    __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
                    bytes = _mm256_and_si256(lowMask, bytes);

                    if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>)
                    {
                        // Subtract zero-point from the integers
                        const uint8_t zp = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b);
                        bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(zp));
                    }
                    else {
                        // Subtract 8 from the integers
                        bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(8));
                    }

                    // Convert to 16-bit int
                    const __m256i vx16_lo =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 0));
                    const __m256i vx16_hi =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 1));

                    // Convert to 32-bit int -> float 32
                    __m512 bvf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo));
                    __m512 bvf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi));
                    bvf_lo = _mm512_mul_ps(bvf_lo, scale_v);
                    bvf_hi = _mm512_mul_ps(bvf_hi, scale_v);

                    acc_lo[nn] = _mm512_fmadd_ps(bvf_lo, av_lo, acc_lo[nn]);
                    acc_lo[nn] = _mm512_fmadd_ps(bvf_hi, av_hi, acc_lo[nn]);

                    b += typename Q4Type::BlobSize;
                }
                b += (MLAS_Q4_N_STRIDE - nblk) * (typename Q4Type::BlobSize);
            }

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = _mm512_reduce_add_ps(acc_lo[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next row
        C += ldc;
        A += lda;
    }
    return CountM;
}

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
    return MlasQ4GemmKernelAvx512f<MLAS_Q4TYPE_BLK1>(CountM, CountN, CountK, C, ldc, Bias, A, lda,
                                                     PackedB);
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
    return MlasQ4GemmKernelAvx512f<MLAS_Q4TYPE_BLK0>(CountM, CountN, CountK, C, ldc, Bias, A, lda,
                                                     PackedB);
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

    const size_t k_blks = MlasDivRoundup(K, (typename Q4TYPE::BlkLen));
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
