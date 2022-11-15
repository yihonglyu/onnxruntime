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
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{32, 512, 2048};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{32, 512, 2048};
};

constexpr size_t MLAS_GEMM_U8S8_KERNEL_AMX::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8S8_KERNEL_AMX::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8S8_KERNEL_AMX::PackedStrides;

extern "C" {

    void
    MLASCALL
    MlasGemmU8S8CopyPackAAmx(
        uint8_t* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        );

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
    MlasGemmU8S8CopyPackAAmx(D, A, lda, CountM, CountK, RowSumBuffer);
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


static inline
void
InitTileWithRowColSumsZeroPoints(
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
    for (size_t m = 0; m < cntM; m++){
        __m512i row;
        if (useZeroPoints){
            row = _mm512_set1_epi32(rowsum_ptr[0]);
            row = _mm512_mullo_epi32(zeropoint, row);
            row = _mm512_maskz_add_epi32(MaskN, colsum, row);
        } else {
            row = _mm512_set1_epi32(rowsum_ptr[0]);
            row = _mm512_maskz_add_epi32(MaskN, colsum, row);
        }
        if (!ZeroMode){
            __m512i c;
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
    MLAS_UNREFERENCED_PARAMETER(ZeroPointB);
    MLAS_UNREFERENCED_PARAMETER(RowSumBuffer);
    MLAS_UNREFERENCED_PARAMETER(ColumnSumBuffer);
    //
    // All 8 tile registers are utilized in the main block.
    // We use Tile 4 - 7 as accumulators, use Tile 2,3 to load
    // 32x64 block from A, and Tile 0,1 to load 64x32 block from B:
    //        B T0  B T1
    //  A T2    T4    T6
    //  A T3    T5    T7
    //
    PackedCountK *= TILE_K;

    size_t m = CountM;
    for (; m >= 2 * TILE_M; m -= 2 * TILE_M) {
        int32_t* c_blk = C; // C - beginning of the row
        int32_t* c16_blk = C + ldc * TILE_M;
        const MLAS_GEMM_U8S8_KERNEL_AMX::PackedBType* b_blk = B; // restart B

        size_t n = CountN;
        for (; n >= 2 * TILE_N; n -= 2 * TILE_N) {
            if (ZeroMode){
                _tile_zero(TMM4);
                _tile_zero(TMM5);
                _tile_zero(TMM6);
                _tile_zero(TMM7);
            } else {
                _tile_loadd(TMM4, c_blk, ldc * sizeof(int32_t));
                _tile_loadd(TMM5, c16_blk, ldc * sizeof(int32_t));
                _tile_loadd(TMM6, c_blk + TILE_N, ldc * sizeof(int32_t));
                _tile_loadd(TMM7, c16_blk + TILE_N, ldc * sizeof(int32_t));
            }

            // Restart A from row start
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_blk = A;
            const MLAS_GEMM_U8S8_KERNEL_AMX::PackedAType* a_next_blk = A + PackedCountK * TILE_M;
            for (size_t k = PackedCountK; k > 0; k -=TILE_K) {
                _tile_loadd(TMM0, b_blk, TILE_K);
                _tile_loadd(TMM2, a_blk, PackedCountK);
                _tile_loadd(TMM3, a_next_blk, PackedCountK);
                _tile_loadd(TMM1, b_blk + PackedCountK * TILE_N, TILE_K);
                _tile_dpbusd(TMM4, TMM2, TMM0);
                _tile_dpbusd(TMM5, TMM3, TMM0);
                _tile_dpbusd(TMM6, TMM2, TMM1);
                _tile_dpbusd(TMM7, TMM3, TMM1);
                b_blk += TILE_N * TILE_K;
                a_blk += TILE_K;
                a_next_blk += TILE_K;
            }
            _tile_stored(TMM4, c_blk, ldc * sizeof(int32_t));
            _tile_stored(TMM5, c16_blk, ldc * sizeof(int32_t));
            _tile_stored(TMM6, c_blk + TILE_N, ldc * sizeof(int32_t));
            _tile_stored(TMM7, c16_blk + TILE_N, ldc * sizeof(int32_t));
            c_blk += 2 * TILE_N;
            c16_blk += 2 * TILE_N;
            b_blk += PackedCountK * TILE_N;
        }

        // Go on to next block of rows
        C += 2 * TILE_M * ldc;
        A += 2 * TILE_M * PackedCountK;
        RowSumBuffer += 2 * TILE_M;
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
