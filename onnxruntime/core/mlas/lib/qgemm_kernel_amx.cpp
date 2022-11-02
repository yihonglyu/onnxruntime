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

#define TILE_M 16
#define TILE_N 16
#define TILE_K 64

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


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
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
    __m512i zmm8;
    zmm8 = _mm512_ternarylogic_epi32(zmm8, zmm8, zmm8, 255);
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
            uint64_t mask = 0xFFFFFFFFFFFFFFFFULL >> (TILE_K - K);
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
            uint64_t mask = 0xFFFFFFFFFFFFFFFFULL >> (TILE_K - K);
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
#pragma GCC diagnostic pop


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


/**
 * @brief Initialize tile accumulator = rowsum + colsum
 * Used when ZeroPointB is null (tensor wide quantization)
*/
static inline
void
InitTileWithRowColSums(
    int32_t*       Tile,
    uint16_t       MaskN,
    const int32_t* rowsum_ptr,
    __m512i        colsum,
    bool           ZeroMode,
    const int32_t* c_blk,
    size_t         ldc
    )
{
    __m512i c0, c1, c2, c3;
    __m512i row0, row1, row2, row3;

    for (size_t cnt = 0; cnt < 4; cnt++){
        row0 = _mm512_set1_epi32(rowsum_ptr[0]);
        row0 = _mm512_maskz_add_epi32(MaskN, colsum, row0);
        row1 = _mm512_set1_epi32(rowsum_ptr[1]);
        row1 = _mm512_maskz_add_epi32(MaskN, colsum, row1);
        row2 = _mm512_set1_epi32(rowsum_ptr[2]);
        row2 = _mm512_maskz_add_epi32(MaskN, colsum, row2);
        row3 = _mm512_set1_epi32(rowsum_ptr[3]);
        row3 = _mm512_maskz_add_epi32(MaskN, colsum, row3);
        if (!ZeroMode){
            c0 = _mm512_maskz_loadu_epi32(MaskN, c_blk + ldc * 0);
            row0 = _mm512_maskz_add_epi32(MaskN, row0, c0);
            c1 = _mm512_maskz_loadu_epi32(MaskN, c_blk + ldc * 1);
            row1 = _mm512_maskz_add_epi32(MaskN, row1, c1);
            c2 = _mm512_maskz_loadu_epi32(MaskN, c_blk + ldc * 2);
            row2 = _mm512_maskz_add_epi32(MaskN, row2, c2);
            c3 = _mm512_maskz_loadu_epi32(MaskN, c_blk + ldc * 3);
            row3 = _mm512_maskz_add_epi32(MaskN, row3, c3);
        }
        _mm512_storeu_si512(Tile + 16 * 0, row0);
        _mm512_storeu_si512(Tile + 16 * 1, row1);
        _mm512_storeu_si512(Tile + 16 * 2, row2);
        _mm512_storeu_si512(Tile + 16 * 3, row3);
        Tile += 16 * 4;
        rowsum_ptr += 4;
        c_blk += ldc * 4;
    }
}


/**
 * @brief Initialize tile accumulator = rowsum * zeropoint + colsum
*/
static inline
void
InitTileWithRowColSumsZeroPoints(
    int32_t*       Tile,
    uint16_t       MaskN,
    const int32_t* rowsum_ptr,
    __m512i        colsum,
    __m512i        zeropoint,
    bool           ZeroMode,
    const int32_t* c_blk,
    size_t         ldc
    )
{
    __m512i c0, c1, c2, c3;
    __m512i row0, row1, row2, row3;

    for (size_t cnt = 0; cnt < 4; cnt++){
        row0 = _mm512_set1_epi32(rowsum_ptr[0]);
        row0 = _mm512_mullo_epi32(zeropoint, row0);
        row1 = _mm512_set1_epi32(rowsum_ptr[1]);
        row1 = _mm512_mullo_epi32(zeropoint, row1);
        row2 = _mm512_set1_epi32(rowsum_ptr[2]);
        row2 = _mm512_mullo_epi32(zeropoint, row2);
        row3 = _mm512_set1_epi32(rowsum_ptr[3]);
        row3 = _mm512_mullo_epi32(zeropoint, row3);
        row0 = _mm512_maskz_add_epi32(MaskN, colsum, row0);
        row1 = _mm512_maskz_add_epi32(MaskN, colsum, row1);
        row2 = _mm512_maskz_add_epi32(MaskN, colsum, row2);
        row3 = _mm512_maskz_add_epi32(MaskN, colsum, row3);
        if (!ZeroMode){
            c0 = _mm512_maskz_loadu_epi32(MaskN, c_blk + ldc * 0);
            row0 = _mm512_maskz_add_epi32(MaskN, row0, c0);
            c1 = _mm512_maskz_loadu_epi32(MaskN, c_blk + ldc * 1);
            row1 = _mm512_maskz_add_epi32(MaskN, row1, c1);
            c2 = _mm512_maskz_loadu_epi32(MaskN, c_blk + ldc * 2);
            row2 = _mm512_maskz_add_epi32(MaskN, row2, c2);
            c3 = _mm512_maskz_loadu_epi32(MaskN, c_blk + ldc * 3);
            row3 = _mm512_maskz_add_epi32(MaskN, row3, c3);
        }
        _mm512_storeu_si512(Tile + 16 * 0, row0);
        _mm512_storeu_si512(Tile + 16 * 1, row1);
        _mm512_storeu_si512(Tile + 16 * 2, row2);
        _mm512_storeu_si512(Tile + 16 * 3, row3);
        Tile += 16 * 4;
        rowsum_ptr += 4;
        c_blk += ldc * 4;
    }
}


static inline
void InitCornerTile(
    int32_t*       Tile,
    size_t         cntM,
    uint16_t       MaskN,
    const int32_t* rowsum_ptr,
    __m512i        colsum,
    bool           useZeroPoints,
    __m512i        zeropoint,
    bool           ZeroMode,
    const int32_t* c_blk,
    size_t         ldc
)
{
    __m512i c;
    __m512i row;
    for (size_t m = 0; m < cntM; m++){
        if (useZeroPoints){
            row = _mm512_set1_epi32(rowsum_ptr[0]);
            row = _mm512_mullo_epi32(zeropoint, row);
            row = _mm512_maskz_add_epi32(MaskN, colsum, row);
        } else {
            row = _mm512_set1_epi32(rowsum_ptr[0]);
            row = _mm512_maskz_add_epi32(MaskN, colsum, row);
        }
        if (!ZeroMode){
            c = _mm512_maskz_loadu_epi32(MaskN, c_blk);
            row = _mm512_maskz_add_epi32(MaskN, row, c);
        }
        _mm512_storeu_si512(Tile, row);
        Tile += 16;
        rowsum_ptr++;
        c_blk += ldc;
    }
}


/**
 * @brief move data from Tile buffer to C
 *
 */
static inline
void
MoveTile(const int32_t* Tile, size_t cntM, uint16_t MaskN, int32_t* c_ptr, size_t ldc)
{
    for (size_t i = 0; i < cntM; i++){
        __m512i c = _mm512_maskz_loadu_epi32(MaskN, Tile);
        Tile += 16;
        _mm512_mask_storeu_epi32(c_ptr, MaskN, c);
        c_ptr += ldc;
    }
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
    //
    // All 8 tile registers are utilized in the main block.
    // We use Tile 4 - 7 as accumulators, use Tile 2,3 to load
    // 32x64 block from A, and Tile 0,1 to load 64x32 block from B:
    //        B T0  B T1
    //  A T2    T4    T6
    //  A T3    T5    T7
    //
    MLAS_DECLSPEC_ALIGN(int32_t Tile4[TILE_M * TILE_N], 64);
    MLAS_DECLSPEC_ALIGN(int32_t Tile5[TILE_M * TILE_N], 64);
    MLAS_DECLSPEC_ALIGN(int32_t Tile6[TILE_M * TILE_N], 64);
    MLAS_DECLSPEC_ALIGN(int32_t Tile7[TILE_M * TILE_N], 64);
    constexpr uint16_t FullMask = 0xFFFF;

    // Compute masks for left over N
    // Values are incorrect when there is no leftover
    auto neg = (0LL - static_cast<int64_t>(CountN)) & (2 * TILE_N - 1);
    const uint32_t dmask = 0xFFFFFFFFUL >> neg;
    const uint16_t MaskN0 = static_cast<uint16_t>(dmask);
    const uint16_t MaskN1 = static_cast<uint16_t>(dmask >> 16);

    size_t m = CountM;
    for (; m >= 2 * TILE_M; m -= 2 * TILE_M) {
        int32_t* c_blk = C; // C - beginning of the row
        const MLAS_GEMM_U8S8_KERNEL_AMX::PackedBType* b_blk = B; // restart B
        const int32_t* col_sum_ptr = ColumnSumBuffer;
        const int32_t* zp_ptr = ZeroPointB;

        size_t n = CountN;
        for (; n >= 2 * TILE_N; n -= 2 * TILE_N) {
            // Restart A from row start
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk = A;

            __m512i colsum = _mm512_loadu_epi32(col_sum_ptr);
            if (ZeroPointB == nullptr){
                InitTileWithRowColSums(
                    Tile4, FullMask, RowSumBuffer, colsum, ZeroMode, c_blk, ldc);
                _tile_loadd(TMM4, Tile4, TILE_N * sizeof(int32_t));
                InitTileWithRowColSums(
                    Tile5, FullMask, RowSumBuffer + TILE_M, colsum,
                    ZeroMode, c_blk + ldc * TILE_M, ldc);
                _tile_loadd(TMM5, Tile5, TILE_N * sizeof(int32_t));
                colsum = _mm512_loadu_epi32(col_sum_ptr + TILE_N);
                InitTileWithRowColSums(
                    Tile6, FullMask, RowSumBuffer, colsum,
                    ZeroMode, c_blk + TILE_N, ldc);
                _tile_loadd(TMM6, Tile6, TILE_N * sizeof(int32_t));
                InitTileWithRowColSums(
                    Tile7, FullMask, RowSumBuffer + TILE_M, colsum,
                    ZeroMode, c_blk + ldc * TILE_M + TILE_N, ldc);
                _tile_loadd(TMM7, Tile7, TILE_N * sizeof(int32_t));
            } else {
                __m512i zeropoint = _mm512_loadu_epi32(zp_ptr);
                InitTileWithRowColSumsZeroPoints(
                    Tile4, FullMask, RowSumBuffer, colsum,
                    zeropoint, ZeroMode, c_blk, ldc);
                _tile_loadd(TMM4, Tile4, TILE_N * sizeof(int32_t));
                InitTileWithRowColSumsZeroPoints(
                    Tile5, FullMask, RowSumBuffer + TILE_M, colsum,
                    zeropoint, ZeroMode, c_blk + ldc * TILE_M, ldc);
                _tile_loadd(TMM5, Tile5, TILE_N * sizeof(int32_t));
                colsum = _mm512_loadu_epi32(col_sum_ptr + TILE_N);
                zeropoint = _mm512_loadu_epi32(zp_ptr + TILE_N);
                InitTileWithRowColSumsZeroPoints(
                    Tile6, FullMask, RowSumBuffer, colsum,
                    zeropoint, ZeroMode, c_blk + TILE_N, ldc);
                _tile_loadd(TMM6, Tile6, TILE_N * sizeof(int32_t));
                InitTileWithRowColSumsZeroPoints(
                    Tile7, FullMask, RowSumBuffer + TILE_M, colsum,
                    zeropoint, ZeroMode, c_blk + ldc * TILE_M + TILE_N, ldc);
                _tile_loadd(TMM7, Tile7, TILE_N * sizeof(int32_t));
                zp_ptr += 2 * TILE_N;
            }
            col_sum_ptr += 2 * TILE_N;

            for (size_t k = 0; k < PackedCountK; k++) {
                _tile_loadd(TMM0, b_blk, TILE_K);
                _tile_loadd(TMM2, a_blk, PackedCountK * TILE_K);
                _tile_loadd(TMM3, a_blk + PackedCountK * TILE_K * TILE_M,
                    PackedCountK * TILE_K);
                _tile_loadd(TMM1, b_blk + PackedCountK * TILE_K * TILE_N, TILE_K);
                _tile_dpbusd(TMM4, TMM2, TMM0);
                _tile_dpbusd(TMM5, TMM3, TMM0);
                _tile_dpbusd(TMM6, TMM2, TMM1);
                _tile_dpbusd(TMM7, TMM3, TMM1);
                b_blk += TILE_N * TILE_K;
                a_blk += TILE_K;
            }
            _tile_stored(TMM4, c_blk, ldc * sizeof(int32_t));
            _tile_stored(TMM5, c_blk + ldc * TILE_M, ldc * sizeof(int32_t));
            _tile_stored(TMM6, c_blk + TILE_N, ldc * sizeof(int32_t));
            _tile_stored(TMM7, c_blk + ldc * TILE_M + TILE_N, ldc * sizeof(int32_t));
            c_blk += 2 * TILE_N;
            b_blk += PackedCountK * TILE_K * TILE_N;
        }

        if (n != 0) {
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk = A;

            __m512i colsum = _mm512_maskz_loadu_epi32(MaskN0, col_sum_ptr);
            if (ZeroPointB == nullptr){
                InitTileWithRowColSums(
                    Tile4, MaskN0, RowSumBuffer, colsum, ZeroMode, c_blk, ldc);
                _tile_loadd(TMM4, Tile4, TILE_N * sizeof(int32_t));
                InitTileWithRowColSums(
                    Tile5, MaskN0, RowSumBuffer + TILE_M, colsum,
                    ZeroMode, c_blk + ldc * TILE_M, ldc);
                _tile_loadd(TMM5, Tile5, TILE_N * sizeof(int32_t));
                if (MaskN1 != 0){
                    colsum = _mm512_maskz_loadu_epi32(MaskN1, col_sum_ptr + TILE_N);
                    InitTileWithRowColSums(
                        Tile6, MaskN1, RowSumBuffer, colsum,
                        ZeroMode, c_blk + TILE_N, ldc);
                    _tile_loadd(TMM6, Tile6, TILE_N * sizeof(int32_t));
                    InitTileWithRowColSums(
                        Tile7, MaskN1, RowSumBuffer + TILE_M, colsum,
                        ZeroMode, c_blk + ldc * TILE_M + TILE_N, ldc);
                    _tile_loadd(TMM7, Tile7, TILE_N * sizeof(int32_t));
                }
            } else {
                __m512i zeropoint = _mm512_maskz_loadu_epi32(MaskN0, zp_ptr);
                InitTileWithRowColSumsZeroPoints(
                    Tile4, MaskN0, RowSumBuffer, colsum,
                    zeropoint, ZeroMode, c_blk, ldc);
                _tile_loadd(TMM4, Tile4, TILE_N * sizeof(int32_t));
                InitTileWithRowColSumsZeroPoints(
                    Tile5, MaskN0, RowSumBuffer + TILE_M, colsum,
                    zeropoint, ZeroMode, c_blk + ldc * TILE_M, ldc);
                _tile_loadd(TMM5, Tile5, TILE_N * sizeof(int32_t));
                if (MaskN1 != 0){
                    colsum = _mm512_maskz_loadu_epi32(MaskN1, col_sum_ptr + TILE_N);
                    zeropoint = _mm512_maskz_loadu_epi32(MaskN1, zp_ptr + TILE_N);
                    InitTileWithRowColSumsZeroPoints(
                        Tile6, MaskN1, RowSumBuffer, colsum,
                        zeropoint, ZeroMode, c_blk + TILE_N, ldc);
                    _tile_loadd(TMM6, Tile6, TILE_N * sizeof(int32_t));
                    InitTileWithRowColSumsZeroPoints(
                        Tile7, MaskN1, RowSumBuffer + TILE_M, colsum,
                        zeropoint, ZeroMode, c_blk + ldc * TILE_M + TILE_N, ldc);
                    _tile_loadd(TMM7, Tile7, TILE_N * sizeof(int32_t));
                }
            }

            for (size_t k = 0; k < PackedCountK; k++) {
                _tile_loadd(TMM0, b_blk, TILE_K);
                _tile_loadd(TMM2, a_blk, PackedCountK * TILE_K);
                _tile_loadd(TMM3, a_blk + PackedCountK * TILE_K * TILE_M,
                    PackedCountK * TILE_K);
                _tile_dpbusd(TMM4, TMM2, TMM0);
                _tile_dpbusd(TMM5, TMM3, TMM0);
                if (MaskN1 != 0){
                    _tile_loadd(TMM1, b_blk + PackedCountK * TILE_K * TILE_N, TILE_K);
                    _tile_dpbusd(TMM6, TMM2, TMM1);
                    _tile_dpbusd(TMM7, TMM3, TMM1);
                }
                b_blk += TILE_N * TILE_K;
                a_blk += TILE_K;
            }
            if ((MaskN0 & 0x8000) != 0){
                _tile_stored(TMM4, c_blk, ldc * sizeof(int32_t));
                _tile_stored(TMM5, c_blk + ldc * TILE_M, ldc * sizeof(int32_t));
            } else {
                _tile_stored(TMM4, Tile4, TILE_N * sizeof(int32_t));
                _tile_stored(TMM5, Tile5, TILE_N * sizeof(int32_t));
                MoveTile(Tile4, TILE_M, MaskN0, c_blk, ldc);
                MoveTile(Tile5, TILE_M, MaskN0, c_blk + ldc * TILE_M, ldc);
            }
            if (MaskN1 != 0){
                _tile_stored(TMM6, Tile6, TILE_N * sizeof(int32_t));
                _tile_stored(TMM7, Tile7, TILE_N * sizeof(int32_t));
                MoveTile(Tile6, TILE_M, MaskN1, c_blk + TILE_N, ldc);
                MoveTile(Tile7, TILE_M, MaskN1, c_blk + ldc * TILE_M + TILE_N, ldc);
            }
        }

        // Go on to next block of rows
        C += 2 * TILE_M * ldc;
        A += 2 * TILE_M * TILE_K * PackedCountK;
        RowSumBuffer += 2 * TILE_M;
    }

    if (m != 0) {
        const int leftover_m = static_cast<int>(m);
        const int m0 = std::min(leftover_m, TILE_M);
        const int m1 = std::max(leftover_m - TILE_M, 0);

        int32_t* c_blk = C; // C - beginning of the row
        const MLAS_GEMM_U8S8_KERNEL_AMX::PackedBType* b_blk = B; // restart B
        const int32_t* col_sum_ptr = ColumnSumBuffer;
        const int32_t* zp_ptr = ZeroPointB;

        size_t n = CountN;
        for (; n >= 2 * TILE_N; n -= 2 * TILE_N) {
            // Restart A from row start
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk = A;

            __m512i zeropoint;
            __m512i colsum = _mm512_loadu_epi32(col_sum_ptr);
            if (ZeroPointB != nullptr){
                zeropoint = _mm512_loadu_epi32(zp_ptr);
            }
            InitCornerTile(Tile4, m0, FullMask, RowSumBuffer, colsum,
                ZeroPointB!=nullptr, zeropoint, ZeroMode, c_blk, ldc);
            _tile_loadd(TMM4, Tile4, TILE_N * sizeof(int32_t));
            if (m1 != 0){
                InitCornerTile(Tile5, m1, FullMask, RowSumBuffer + TILE_M, colsum,
                    ZeroPointB!=nullptr, zeropoint, ZeroMode, c_blk + ldc * TILE_M, ldc);
                _tile_loadd(TMM5, Tile5, TILE_N * sizeof(int32_t));
            }
            colsum = _mm512_loadu_epi32(col_sum_ptr + TILE_N);
            if (ZeroPointB != nullptr) {
                zeropoint = _mm512_loadu_epi32(zp_ptr + TILE_N);
                zp_ptr += 2 * TILE_N;
            }
            InitCornerTile(Tile6, m0, FullMask, RowSumBuffer, colsum,
                ZeroPointB!=nullptr, zeropoint, ZeroMode, c_blk + TILE_N, ldc);
            _tile_loadd(TMM6, Tile6, TILE_N * sizeof(int32_t));
            if (m1 != 0){
                InitCornerTile(Tile7, m1, FullMask, RowSumBuffer + TILE_M, colsum,
                    ZeroPointB!=nullptr, zeropoint, ZeroMode, c_blk + ldc * TILE_M + TILE_N, ldc);
                _tile_loadd(TMM7, Tile7, TILE_N * sizeof(int32_t));
            }
            col_sum_ptr += 2 * TILE_N;

            for (size_t k = 0; k < PackedCountK; k++) {
                _tile_loadd(TMM0, b_blk, TILE_K);
                _tile_loadd(TMM2, a_blk, PackedCountK * TILE_K);
                _tile_loadd(TMM1, b_blk + PackedCountK * TILE_K * TILE_N, TILE_K);
                _tile_dpbusd(TMM4, TMM2, TMM0);
                _tile_dpbusd(TMM6, TMM2, TMM1);
                if (m1 > 0){
                    _tile_loadd(TMM3, a_blk + PackedCountK * TILE_K * TILE_M,
                        PackedCountK * TILE_K);
                    _tile_dpbusd(TMM5, TMM3, TMM0);
                    _tile_dpbusd(TMM7, TMM3, TMM1);
                }
                b_blk += TILE_N * TILE_K;
                a_blk += TILE_K;
            }
            if (m0 == TILE_M) {
                _tile_stored(TMM4, c_blk, ldc * sizeof(int32_t));
                _tile_stored(TMM6, c_blk + TILE_N, ldc * sizeof(int32_t));
            } else {
                _tile_stored(TMM4, Tile4, TILE_N * sizeof(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * sizeof(int32_t));
                MoveTile(Tile4, m0, FullMask, c_blk, ldc);
                MoveTile(Tile6, m0, FullMask, c_blk + TILE_N, ldc);
            }
            if (m1 != 0){
                _tile_stored(TMM5, Tile5, TILE_N * sizeof(int32_t));
                MoveTile(Tile5, m1, FullMask, c_blk + ldc * TILE_M, ldc);
                _tile_stored(TMM7, Tile7, TILE_N * sizeof(int32_t));
                MoveTile(Tile7, m1, FullMask, c_blk + ldc * TILE_M + TILE_N, ldc);
            }
            c_blk += 2 * TILE_N;
            b_blk += PackedCountK * TILE_K * TILE_N;
        }

        if (n != 0) {
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk = A;

            // Init accumulator tiles
            //        B T0  B T1
            //  A T2    T4    T6
            //  A T3    T5    T7
            __m512i zeropoint;
            __m512i colsum = _mm512_maskz_loadu_epi32(MaskN0, col_sum_ptr);
            if (ZeroPointB != nullptr){
                zeropoint = _mm512_maskz_loadu_epi32(MaskN0, zp_ptr);
            }
            InitCornerTile(Tile4, m0, MaskN0, RowSumBuffer, colsum,
                ZeroPointB!=nullptr, zeropoint, ZeroMode, c_blk, ldc);
            _tile_loadd(TMM4, Tile4, TILE_N * sizeof(int32_t));
            if (m1 > 0){
                InitCornerTile(Tile5, m1, MaskN0, RowSumBuffer + TILE_M, colsum,
                    ZeroPointB!=nullptr, zeropoint, ZeroMode, c_blk + ldc * TILE_M, ldc);
                _tile_loadd(TMM5, Tile5, TILE_N * sizeof(int32_t));
            }
            if (MaskN1 != 0){
                colsum = _mm512_maskz_loadu_epi32(MaskN1, col_sum_ptr + TILE_N);
                if (ZeroPointB!=nullptr){
                    zeropoint = _mm512_maskz_loadu_epi32(MaskN1, zp_ptr + TILE_N);
                }
                InitCornerTile(Tile6, m0, MaskN1, RowSumBuffer, colsum,
                    ZeroPointB!=nullptr, zeropoint, ZeroMode, c_blk + TILE_N, ldc);
                _tile_loadd(TMM6, Tile6, TILE_N * sizeof(int32_t));
                if (m1 > 0){
                    InitCornerTile(Tile7, m1, MaskN1, RowSumBuffer + TILE_M, colsum,
                        ZeroPointB!=nullptr, zeropoint, ZeroMode, c_blk + ldc * TILE_M + TILE_N, ldc);
                    _tile_loadd(TMM7, Tile7, TILE_N * sizeof(int32_t));
                }
            }

            for (size_t k = 0; k < PackedCountK; k++) {
                _tile_loadd(TMM0, b_blk, static_cast<int>(TILE_K));
                _tile_loadd(TMM2, a_blk, static_cast<int>(PackedCountK * TILE_K));
                _tile_dpbusd(TMM4, TMM2, TMM0);
                if (m1 > 0){
                    _tile_loadd(TMM3, a_blk + PackedCountK * TILE_K * TILE_M, PackedCountK * TILE_K);
                    _tile_dpbusd(TMM5, TMM3, TMM0);
                }
                if (MaskN1 != 0){
                    _tile_loadd(TMM1, b_blk + PackedCountK * TILE_K * TILE_N, TILE_K);
                    _tile_dpbusd(TMM6, TMM2, TMM1);
                    if (m1 > 0){
                        _tile_dpbusd(TMM7, TMM3, TMM1);
                    }
                }
                b_blk += TILE_N * TILE_K;
                a_blk += TILE_K;
            }
            if ((MaskN0 & 0x8000) != 0 && m0 == TILE_M){
                _tile_stored(TMM4, c_blk, ldc * sizeof(int32_t));
            } else {
                _tile_stored(TMM4, Tile4, TILE_N * sizeof(int32_t));
                MoveTile(Tile4, m0, MaskN0, c_blk, ldc);
            }
            if (m1 > 0){
                _tile_stored(TMM5, Tile5, TILE_N * sizeof(int32_t));
                MoveTile(Tile5, m1, MaskN0, c_blk + ldc * TILE_M, ldc);
            }
            if (MaskN1 != 0){
                _tile_stored(TMM6, Tile6, TILE_N * sizeof(int32_t));
                MoveTile(Tile6, m0, MaskN1, c_blk + TILE_N, ldc);
                if (m1 > 0){
                    _tile_stored(TMM7, Tile7, TILE_N * sizeof(int32_t));
                    MoveTile(Tile7, m1, MaskN1, c_blk + ldc * TILE_M + TILE_N, ldc);
                }
            }
        }
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
