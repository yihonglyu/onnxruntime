/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_amx.cpp

Abstract:

    This module implements QGEMM kernels for amx.

--*/

#include "mlasi.h"
#include "qgemm.h"


#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

#define KPACK (4 / sizeof(type_t))  // Vertical K packing into Dword

constexpr size_t TILE_M = 16;
constexpr size_t TILE_N = 16;
constexpr size_t TILE_K = 64;

/*******************************************************************
 * Packing and Gemm kernels for U8S8 AMX
 ******************************************************************/
struct MLAS_GEMM_U8S8_KERNEL_AMX {
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef int8_t  OffsetBType;

    static constexpr size_t PackedK = TILE_K;

    // Use smaller stride for debugging,
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{128, 128, 128};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{128, 128, 128};
};

constexpr size_t MLAS_GEMM_U8S8_KERNEL_AMX::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8S8_KERNEL_AMX::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8S8_KERNEL_AMX::PackedStrides;

extern "C" {

    void
    MLASCALL
    MlasGemmU8S8CopyPackBAmx(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumBuffer,
        bool BIsSigned
        );

}


template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_U8S8_KERNEL_AMX>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    if (AIsSigned) {
        ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    }

    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_U8S8_KERNEL_AMX>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8S8_KERNEL_AMX::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}


template<>
MLAS_FORCEINLINE
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8S8_KERNEL_AMX>(
    MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    //
    // The packed buffer has the same data ordering as the source bytes,
    // but CountK is aligned up to a multiple of 64 to fill AMX tiles.
    // All extra bytes are zero-padded.
    //
    const size_t AlignedK = (CountK + TILE_K - 1) / TILE_K * TILE_K;
    __m512i zmm8 = _mm512_ternarylogic_epi32(zmm8, zmm8, zmm8, 255);
    zmm8 = _mm512_srli_epi16(zmm8, 15);         // 0x0001;
    __m512i zmm9 = _mm512_slli_epi16(zmm8, 8);
    zmm9 = _mm512_or_epi32(zmm8, zmm9);         // 0x0101
    __m512i zmm10 = _mm512_setzero_epi32();

    for (; CountM >= 4; CountM -= 4){
        // Init row sum accumulators
        __m512i zmm0 = _mm512_setzero_epi32();
        __m512i zmm1 = _mm512_setzero_epi32();
        __m512i zmm2 = _mm512_setzero_epi32();
        __m512i zmm3 = _mm512_setzero_epi32();

        const uint8_t* src_blk = A; // start of the row
        const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* dst_blk = D;
        A += lda * 4;
        D += AlignedK * 4;

        size_t K = CountK;
        for (; K >= TILE_K; K -= TILE_K, src_blk += TILE_K, dst_blk += TILE_K){
            // Load 4 rows
            __m512i zmm4 = _mm512_loadu_si512((void*)src_blk);
            __m512i zmm5 = _mm512_loadu_si512((void*)(src_blk + lda));
            __m512i zmm6 = _mm512_loadu_si512((void*)(src_blk + lda * 2));
            __m512i zmm7 = _mm512_loadu_si512((void*)(src_blk + lda * 3));

            // Store 4 rows with the same layout
            _mm512_store_epi64((void*)(dst_blk), zmm4);
            _mm512_store_epi64((void*)(dst_blk + AlignedK), zmm5);
            _mm512_store_epi64((void*)(dst_blk + AlignedK * 2), zmm6);
            _mm512_store_epi64((void*)(dst_blk + AlignedK * 3), zmm7);

            // Row sums -> 32b accumulators
            // TODO maybe use 16b accumulator, save to 32b every 256 bytes?
            zmm4 = _mm512_maddubs_epi16(zmm4, zmm9); // byte + byte -> short
            zmm4 = _mm512_madd_epi16(zmm4, zmm8);    // short + short -> int32
            zmm0 = _mm512_add_epi32(zmm0, zmm4);
            zmm5 = _mm512_maddubs_epi16(zmm5, zmm9); // byte + byte -> short
            zmm5 = _mm512_madd_epi16(zmm5, zmm8);    // short + short -> int32
            zmm1 = _mm512_add_epi32(zmm1, zmm5);
            zmm6 = _mm512_maddubs_epi16(zmm6, zmm9); // byte + byte -> short
            zmm6 = _mm512_madd_epi16(zmm6, zmm8);    // short + short -> int32
            zmm2 = _mm512_add_epi32(zmm2, zmm6);
            zmm7 = _mm512_maddubs_epi16(zmm7, zmm9); // byte + byte -> short
            zmm7 = _mm512_madd_epi16(zmm7, zmm8);    // short + short -> int32
            zmm3 = _mm512_add_epi32(zmm3, zmm7);
        }

        if (K > 0){
            // process remaining columns
            uint64_t mask = 0xFFFFFFFFFFFFFFFF >> (TILE_K - K);
            __m512i zmm4 = _mm512_mask_loadu_epi8(zmm10, mask, (void*)src_blk);
            __m512i zmm5 = _mm512_mask_loadu_epi8(zmm10, mask, (void*)(src_blk + lda));
            __m512i zmm6 = _mm512_mask_loadu_epi8(zmm10, mask, (void*)(src_blk + lda * 2));
            __m512i zmm7 = _mm512_mask_loadu_epi8(zmm10, mask, (void*)(src_blk + lda * 3));
            _mm512_store_epi64((void*)(dst_blk), zmm4);
            _mm512_store_epi64((void*)(dst_blk + AlignedK), zmm5);
            _mm512_store_epi64((void*)(dst_blk + AlignedK * 2), zmm6);
            _mm512_store_epi64((void*)(dst_blk + AlignedK * 3), zmm7);
            zmm4 = _mm512_maddubs_epi16(zmm4, zmm9); // byte + byte -> short
            zmm4 = _mm512_madd_epi16(zmm4, zmm8);    // short + short -> int32
            zmm0 = _mm512_add_epi32(zmm0, zmm4);
            zmm5 = _mm512_maddubs_epi16(zmm5, zmm9); // byte + byte -> short
            zmm5 = _mm512_madd_epi16(zmm5, zmm8);    // short + short -> int32
            zmm1 = _mm512_add_epi32(zmm1, zmm5);
            zmm6 = _mm512_maddubs_epi16(zmm6, zmm9); // byte + byte -> short
            zmm6 = _mm512_madd_epi16(zmm6, zmm8);    // short + short -> int32
            zmm2 = _mm512_add_epi32(zmm2, zmm6);
            zmm7 = _mm512_maddubs_epi16(zmm7, zmm9); // byte + byte -> short
            zmm7 = _mm512_madd_epi16(zmm7, zmm8);    // short + short -> int32
            zmm3 = _mm512_add_epi32(zmm3, zmm7);
        }

        // Reduce row sums
        __m256i ymm0  = _mm512_castsi512_si256(zmm0);
        __m256i high0 = _mm512_extracti64x4_epi64(zmm0,1);
        ymm0 = _mm256_add_epi32(ymm0, high0);
        __m256i ymm1  = _mm512_castsi512_si256(zmm1);
        __m256i high1 = _mm512_extracti64x4_epi64(zmm1,1);
        ymm1 = _mm256_add_epi32(ymm1, high1);
        ymm0 = _mm256_hadd_epi32(ymm0, ymm1); // reduce and interleave Sum1/Sum0
        __m256i ymm2  = _mm512_castsi512_si256(zmm2);
        __m256i high2 = _mm512_extracti64x4_epi64(zmm2,1);
        ymm2 = _mm256_add_epi32(ymm2, high2);
        __m256i ymm3  = _mm512_castsi512_si256(zmm3);
        __m256i high3 = _mm512_extracti64x4_epi64(zmm3,1);
        ymm3 = _mm256_add_epi32(ymm3, high3);
        ymm1 = _mm256_hadd_epi32(ymm2, ymm3); // reduce and interleave Sum3/Sum2
        ymm0 = _mm256_hadd_epi32(ymm0, ymm1); // reduce and interleave Sum3/Sum2/Sum1/Sum0
        __m128i xmm0 = _mm256_castsi256_si128(ymm0);
        __m128i xmm1 = _mm256_extracti128_si256(ymm0, 1);
        xmm0 =  _mm_add_epi32(xmm0, xmm1);   // reduce low/high dwords
        _mm_store_epi32((void*)RowSumBuffer, xmm0);
        RowSumBuffer += 4;
    }

    if (CountM >= 2){
        CountM -= 2;
        // Init row sum accumulators
        __m512i zmm0 = _mm512_setzero_epi32();
        __m512i zmm1 = _mm512_setzero_epi32();

        const uint8_t* src_blk = A; // start of the row
        const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* dst_blk = D;
        A += lda * 2;
        D += AlignedK * 2;

        size_t K = CountK;
        for (; K >= TILE_K; K -= TILE_K, src_blk += TILE_K, dst_blk += TILE_K){
            __m512i zmm4 = _mm512_loadu_si512((void*)src_blk);
            __m512i zmm5 = _mm512_loadu_si512((void*)(src_blk + lda));
            _mm512_store_epi64((void*)(dst_blk), zmm4);
            _mm512_store_epi64((void*)(dst_blk + AlignedK), zmm5);
            zmm4 = _mm512_maddubs_epi16(zmm4, zmm9); // byte + byte -> short
            zmm4 = _mm512_madd_epi16(zmm4, zmm8);    // short + short -> int32
            zmm0 = _mm512_add_epi32(zmm0, zmm4);
            zmm5 = _mm512_maddubs_epi16(zmm5, zmm9); // byte + byte -> short
            zmm5 = _mm512_madd_epi16(zmm5, zmm8);    // short + short -> int32
            zmm1 = _mm512_add_epi32(zmm1, zmm5);
        }

        if (K > 0){
            // process remaining columns
            uint64_t mask = 0xFFFFFFFFFFFFFFFF >> (TILE_K - K);
            __m512i zmm4 = _mm512_mask_loadu_epi8(zmm10, mask, (void*)src_blk);
            __m512i zmm5 = _mm512_mask_loadu_epi8(zmm10, mask, (void*)(src_blk + lda));
            _mm512_store_epi64((void*)(dst_blk), zmm4);
            _mm512_store_epi64((void*)(dst_blk + AlignedK), zmm5);
            zmm4 = _mm512_maddubs_epi16(zmm4, zmm9); // byte + byte -> short
            zmm4 = _mm512_madd_epi16(zmm4, zmm8);    // short + short -> int32
            zmm0 = _mm512_add_epi32(zmm0, zmm4);
            zmm5 = _mm512_maddubs_epi16(zmm5, zmm9); // byte + byte -> short
            zmm5 = _mm512_madd_epi16(zmm5, zmm8);    // short + short -> int32
            zmm1 = _mm512_add_epi32(zmm1, zmm5);
        }

        // Reduce row sums
        int32_t sum0 = _mm512_reduce_add_epi32(zmm0);
        int32_t sum1 = _mm512_reduce_add_epi32(zmm1);
        RowSumBuffer[0] = sum0;
        RowSumBuffer[1] = sum1;
        RowSumBuffer+=2;
    }

    if (CountM > 0){
        __m512i zmm0 = _mm512_setzero_epi32();

        const uint8_t* src_blk = A; // start of the row
        const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* dst_blk = D;
        A += lda;
        D += AlignedK;

        size_t K = CountK;
        for (; K >= TILE_K; K -= TILE_K, src_blk += TILE_K, dst_blk += TILE_K){
            __m512i zmm4 = _mm512_loadu_si512((void*)src_blk);
            _mm512_store_epi64((void*)(dst_blk), zmm4);
            zmm4 = _mm512_maddubs_epi16(zmm4, zmm9); // byte + byte -> short
            zmm4 = _mm512_madd_epi16(zmm4, zmm8);    // short + short -> int32
            zmm0 = _mm512_add_epi32(zmm0, zmm4);
        }

        if (K > 0){
            // process remaining columns
            uint64_t mask = 0xFFFFFFFFFFFFFFFF >> (TILE_K - K);
            __m512i zmm4 = _mm512_mask_loadu_epi8(zmm10, mask, (void*)src_blk);
            _mm512_store_epi64((void*)(dst_blk), zmm4);
            zmm4 = _mm512_maddubs_epi16(zmm4, zmm9); // byte + byte -> short
            zmm4 = _mm512_madd_epi16(zmm4, zmm8);    // short + short -> int32
            zmm0 = _mm512_add_epi32(zmm0, zmm4);
        }

        // Reduce row sums
        int32_t sum = _mm512_reduce_add_epi32(zmm0);
        *RowSumBuffer = sum;
        RowSumBuffer++;
    }
}


template<>
MLAS_FORCEINLINE
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8S8_KERNEL_AMX>(
    MLAS_GEMM_U8S8_KERNEL_AMX::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    MlasGemmU8S8CopyPackBAmx(D, B, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
}


template <>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U8S8_KERNEL_AMX>(
    const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* A,
    const MLAS_GEMM_U8S8_KERNEL_AMX::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode)
{
    MLAS_UNREFERENCED_PARAMETER(RowSumBuffer);
    MLAS_UNREFERENCED_PARAMETER(ColumnSumBuffer);
    MLAS_UNREFERENCED_PARAMETER(ZeroPointB);

    const size_t K = PackedCountK * MLAS_GEMM_U8S8_KERNEL_AMX::PackedK;
    const int cstride = static_cast<int>(ldc * sizeof(int32_t));

    size_t m = CountM;
    while (m >= 2 * TILE_M) {
        m -= 2 * TILE_M;
        int32_t* c_blk_ptr = C; // C - beginning of the row
        const MLAS_GEMM_U8S8_KERNEL_AMX::PackedBType* b_blk_ptr = B; // restart B

        size_t n = CountN;
        while (n >= 2 * TILE_N) {
            n -= 2 * TILE_N;
            // Restart A from row start
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk_ptr = A;

            // Init accumulator tiles
            //        B T0  B T1
            //  A T2    T4    T6
            //  A T3    T5    T7
            if (ZeroMode) {
                _tile_zero(TMM4);
                _tile_zero(TMM5);
                _tile_zero(TMM6);
                _tile_zero(TMM7);
            } else {
                _tile_loadd(TMM4, (void*)c_blk_ptr, cstride);
                _tile_loadd(TMM5, (void*)(c_blk_ptr + TILE_M * ldc), cstride);
                _tile_loadd(TMM6, (void*)(c_blk_ptr + TILE_N), cstride);
                _tile_loadd(TMM7, (void*)(c_blk_ptr + TILE_M * ldc + TILE_N), cstride);
            }

            for (size_t k = 0; k < K; k += TILE_K) {
                _tile_loadd(TMM0, (void*)b_blk_ptr, static_cast<int>(64));
                _tile_loadd(TMM2, (void*)a_blk_ptr, static_cast<int>(K));
                _tile_dpbusd(TMM4, TMM2, TMM0);
                _tile_loadd(TMM3, (void*)(a_blk_ptr + TILE_M * K), static_cast<int>(K));
                _tile_dpbusd(TMM5, TMM3, TMM0);
                _tile_loadd(TMM1, (void*)(b_blk_ptr + TILE_N * K), static_cast<int>(64));
                _tile_dpbusd(TMM6, TMM2, TMM1);
                _tile_dpbusd(TMM7, TMM3, TMM1);
                b_blk_ptr += TILE_N * TILE_K;
                a_blk_ptr += TILE_K;
            }
            _tile_stored(TMM4, (void*)c_blk_ptr, cstride);
            _tile_stored(TMM5, (void*)(c_blk_ptr + TILE_M * ldc), cstride);
            _tile_stored(TMM6, (void*)(c_blk_ptr + TILE_N), cstride);
            _tile_stored(TMM7, (void*)(c_blk_ptr + TILE_M * ldc + TILE_N), cstride);
            c_blk_ptr += 2 * TILE_N;
            b_blk_ptr += K * TILE_N;
        }

        if (n == TILE_N) {
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk_ptr = A;

            // Init accumulator tiles
            //        B T0
            //  A T2    T4
            //  A T3    T5
            if (ZeroMode) {
                _tile_zero(TMM4);
                _tile_zero(TMM5);
            } else {
                _tile_loadd(TMM4, (void*)c_blk_ptr, cstride);
                _tile_loadd(TMM5, (void*)(c_blk_ptr + TILE_M * ldc), cstride);
            }

            for (size_t k = 0; k < K; k += TILE_K) {
                _tile_loadd(TMM0, (void*)b_blk_ptr, static_cast<int>(64));
                _tile_loadd(TMM2, (void*)a_blk_ptr, static_cast<int>(K));
                _tile_dpbusd(TMM4, TMM2, TMM0);
                _tile_loadd(TMM3, (void*)(a_blk_ptr + TILE_M * K), static_cast<int>(K));
                _tile_dpbusd(TMM5, TMM3, TMM0);
                b_blk_ptr += TILE_N * TILE_K;
                a_blk_ptr += TILE_K;
            }
            _tile_stored(TMM4, (void*)c_blk_ptr, cstride);
            _tile_stored(TMM5, (void*)(c_blk_ptr + TILE_M * ldc), cstride);
            c_blk_ptr += TILE_N;
        } else if (n!=0){
            throw new std::runtime_error("Leftover columns not handled in AMX kernel yet!");
        }

        // Go on to next block of rows
        C += 2 * TILE_M * ldc; // points to beginning of the rows
        A += 2 * TILE_M * K;
    }

    if (m == TILE_M) {
        int32_t* c_blk_ptr = C; // C - beginning of the row
        const MLAS_GEMM_U8S8_KERNEL_AMX::PackedBType* b_blk_ptr = B; // restart B
        size_t n = CountN;
        while (n >= 2 * TILE_N) {
            n -= 2 * TILE_N;

            // Restart A from row start
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk_ptr = A;

            // Init accumulator tiles
            //        B T0  B T1
            //  A T2    T4    T6
            //  A T3    T5    T7
            if (ZeroMode) {
                _tile_zero(TMM4);
                _tile_zero(TMM6);
            } else {
                _tile_loadd(TMM4, (void*)c_blk_ptr, cstride);
                _tile_loadd(TMM6, (void*)(c_blk_ptr + TILE_N), cstride);
            }

            for (size_t k = 0; k < K; k += TILE_K) {
                _tile_loadd(TMM0, (void*)b_blk_ptr, static_cast<int>(64));
                _tile_loadd(TMM2, (void*)a_blk_ptr, static_cast<int>(K));
                _tile_dpbusd(TMM4, TMM2, TMM0);
                _tile_loadd(TMM1, (void*)(b_blk_ptr + K * TILE_N), static_cast<int>(64));
                _tile_dpbusd(TMM6, TMM2, TMM1);
                b_blk_ptr += TILE_N * TILE_K;
                a_blk_ptr += TILE_K;
            }
            _tile_stored(TMM4, (void*)c_blk_ptr, cstride);
            _tile_stored(TMM6, (void*)(c_blk_ptr + TILE_N), cstride);
            c_blk_ptr += 2 * TILE_N;
            b_blk_ptr += K * TILE_N;
        }

        if (n == TILE_N) {
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk_ptr = A;

            // Init accumulator tiles
            if (ZeroMode) {
                _tile_zero(TMM4);
            } else {
                _tile_loadd(TMM4, (void*)c_blk_ptr, cstride);
            }

            for (size_t k = 0; k < K; k += TILE_K) {
                _tile_loadd(TMM0, (void*)b_blk_ptr, static_cast<int>(64));
                _tile_loadd(TMM2, (void*)a_blk_ptr, static_cast<int>(K));
                _tile_dpbusd(TMM4, TMM2, TMM0);
                b_blk_ptr += TILE_N * TILE_K;
                a_blk_ptr += TILE_K;
            }
            _tile_stored(TMM4, (void*)c_blk_ptr, cstride);
            c_blk_ptr += TILE_N;
        } else if (n!=0){
            throw new std::runtime_error("Leftover columns not handled in AMX kernel yet!");
        }

        // Go on to next block of rows
        C += TILE_M * ldc; // points to beginning of the rows
        A += TILE_M * K;
    } else if (m > 0) {
        throw new std::runtime_error("Leftover rows not handled in AMX kernel yet!");
    }

    return CountM;
}


const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8S8DispatchAmx = {
    MlasGemmQuantOperation<MLAS_GEMM_U8S8_KERNEL_AMX>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_U8S8_KERNEL_AMX>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_U8S8_KERNEL_AMX>,
    MLAS_GEMM_U8S8_KERNEL_AMX::PackedK,
    MLAS_GEMM_U8S8_KERNEL_AMX::PackedStrides.K,
    64  // StridM
};
