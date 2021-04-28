/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize_avx512f.cpp

Abstract:

    This module implements routines to quantize buffers with AVX512F instructions.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "mlasi.h"

#ifndef _MM_K0_REG16
#define _MM_K0_REG16 0xffff
#endif

//
// QuantizeLinear implementation using AVX512 intrinsics.
//

template <typename OutputType>
void
MLASCALL
MlasQuantizeLinearAvx512F(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters with AVX512 instructions.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    auto ScaleVector = _mm512_set1_ps(Scale);
    auto MinimumValueVector = _mm512_set1_ps(float(MinimumValue - ZeroPoint));
    auto MaximumValueVector = _mm512_set1_ps(float(MaximumValue - ZeroPoint));
    auto ZeroPointVector = _mm512_set1_epi32(ZeroPoint);

    while (N >= 64) {

        auto FloatVector0 = _mm512_loadu_ps(Input);
        auto FloatVector1 = _mm512_loadu_ps(Input + 16);
        auto FloatVector2 = _mm512_loadu_ps(Input + 32);
        auto FloatVector3 = _mm512_loadu_ps(Input + 48);

        FloatVector0 = _mm512_div_ps(FloatVector0, ScaleVector);
        FloatVector1 = _mm512_div_ps(FloatVector1, ScaleVector);
        FloatVector2 = _mm512_div_ps(FloatVector2, ScaleVector);
        FloatVector3 = _mm512_div_ps(FloatVector3, ScaleVector);

        FloatVector0 = _mm512_max_ps(FloatVector0, MinimumValueVector);
        FloatVector1 = _mm512_max_ps(FloatVector1, MinimumValueVector);
        FloatVector2 = _mm512_max_ps(FloatVector2, MinimumValueVector);
        FloatVector3 = _mm512_max_ps(FloatVector3, MinimumValueVector);

        FloatVector0 = _mm512_min_ps(FloatVector0, MaximumValueVector);
        FloatVector1 = _mm512_min_ps(FloatVector1, MaximumValueVector);
        FloatVector2 = _mm512_min_ps(FloatVector2, MaximumValueVector);
        FloatVector3 = _mm512_min_ps(FloatVector3, MaximumValueVector);

        auto IntegerVector0 = _mm512_cvtps_epi32(FloatVector0);
        auto IntegerVector1 = _mm512_cvtps_epi32(FloatVector1);
        auto IntegerVector2 = _mm512_cvtps_epi32(FloatVector2);
        auto IntegerVector3 = _mm512_cvtps_epi32(FloatVector3);

        IntegerVector0 = _mm512_add_epi32(IntegerVector0, ZeroPointVector);
        IntegerVector1 = _mm512_add_epi32(IntegerVector1, ZeroPointVector);
        IntegerVector2 = _mm512_add_epi32(IntegerVector2, ZeroPointVector);
        IntegerVector3 = _mm512_add_epi32(IntegerVector3, ZeroPointVector);

        _mm512_mask_cvtepi32_storeu_epi8(Output, _MM_K0_REG16, IntegerVector0);
        _mm512_mask_cvtepi32_storeu_epi8(Output + 16, _MM_K0_REG16, IntegerVector1);
        _mm512_mask_cvtepi32_storeu_epi8(Output + 32, _MM_K0_REG16, IntegerVector2);
        _mm512_mask_cvtepi32_storeu_epi8(Output + 48, _MM_K0_REG16, IntegerVector3);

        Input += 64;
        Output += 64;
        N -= 64;
    }

    while (N >= 16) {
        auto FloatVector = _mm512_loadu_ps(Input);
        FloatVector = _mm512_div_ps(FloatVector, ScaleVector);
        FloatVector = _mm512_max_ps(FloatVector, MinimumValueVector);
        FloatVector = _mm512_min_ps(FloatVector, MaximumValueVector);

        auto IntegerVector = _mm512_cvtps_epi32(FloatVector);
        IntegerVector = _mm512_add_epi32(IntegerVector, ZeroPointVector);

        _mm512_mask_cvtepi32_storeu_epi8(Output, _MM_K0_REG16, IntegerVector);

        Input += 16;
        Output += 16;
        N -= 16;
    }

    if (N > 0) {
        __mmask16 mask = uint16_t((uint32_t(1) << N) - uint32_t(1));
        auto FloatVector = _mm512_maskz_loadu_ps(mask, Input);
        FloatVector = _mm512_div_ps(FloatVector, ScaleVector);
        FloatVector = _mm512_max_ps(FloatVector, MinimumValueVector);
        FloatVector = _mm512_min_ps(FloatVector, MaximumValueVector);

        auto IntegerVector = _mm512_cvtps_epi32(FloatVector);
        IntegerVector = _mm512_add_epi32(IntegerVector, ZeroPointVector);

        _mm512_mask_cvtepi32_storeu_epi8(Output, mask, IntegerVector);
    }
}

void
MLASCALL
MlasQuantizeLinearU8KernelAvx512F(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
{
    MlasQuantizeLinearAvx512F<uint8_t>(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasQuantizeLinearS8KernelAvx512F(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
    MlasQuantizeLinearAvx512F<int8_t>(Input, Output, N, Scale, ZeroPoint);
}

template<typename OutputType>
MLAS_FORCEINLINE void
MlasMaskedPackStore(OutputType* Output, __mmask16 mask, __m512i IntegerVector);

template<>
MLAS_FORCEINLINE void
MlasMaskedPackStore(int16_t* Output, __mmask16 mask, __m512i IntegerVector)
{
    _mm512_mask_cvtepi32_storeu_epi16(Output, mask, IntegerVector);
}

template<>
MLAS_FORCEINLINE void
MlasMaskedPackStore(uint8_t* Output, __mmask16 mask, __m512i IntegerVector)
{
    _mm512_mask_cvtepi32_storeu_epi8(Output, mask, IntegerVector);
}

/**
 * @brief Quantize and prepack a block of A, AVX512F implementation
 *
 * Type parameter OutputType can be either uint8_t or int16_t, because
 * some AVX QGEMM kernels need to extend A to 16b and others keep 8b.
 * Since quantized A is always uint8_t, extension to 16b is simple
 * zero extension
 *
 * @param D         Destination buffer for quantized and packed values
 * @param A         Buffer for floating matrix A
 * @param lda       leading dimension for A
 * @param CountM    Number of rows to process
 * @param CountK    Number of columns to process
 * @param RowSumBuffer
 * @param Scale     Quantization scale
 * @param ZeroPoint Quantization zero point value
 */
template<typename OutputType>
MLAS_FORCEINLINE
void
MlasQuantizeLinearPackAAvx512F(
    OutputType* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    float Scale,
    uint8_t ZeroPoint
    )
{
    // For SSE and AVX kernels, pack A does not do much other than align each row
    // to 32 bits boundary and pad with 0
    constexpr size_t kbound = 4 / sizeof(OutputType);
    const size_t AlignedK = (CountK + kbound - 1) & ~(kbound - 1);

    // const values needed for the entire matrix
    constexpr int32_t MinimumValue = std::numeric_limits<uint8_t>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<uint8_t>::max();

    auto ScaleVector = _mm512_set1_ps(Scale);
    auto MinimumValueVector = _mm512_set1_ps(float(MinimumValue - ZeroPoint));
    auto MaximumValueVector = _mm512_set1_ps(float(MaximumValue - ZeroPoint));
    auto ZeroPointVector = _mm512_set1_epi32(ZeroPoint);
    
    while (CountM > 0) {

        const float* a = A;
        size_t k = CountK;

        // Row accumulators
        auto AccVector0 = _mm512_setzero_epi32();
        auto AccVector1 = _mm512_setzero_epi32();
        auto AccVector2 = _mm512_setzero_epi32();
        auto AccVector3 = _mm512_setzero_epi32();

        while (k >= 64) {
            auto FloatVector0 = _mm512_loadu_ps(a);
            auto FloatVector1 = _mm512_loadu_ps(a + 16);
            auto FloatVector2 = _mm512_loadu_ps(a + 32);
            auto FloatVector3 = _mm512_loadu_ps(a + 48);

            FloatVector0 = _mm512_div_ps(FloatVector0, ScaleVector);
            FloatVector1 = _mm512_div_ps(FloatVector1, ScaleVector);
            FloatVector2 = _mm512_div_ps(FloatVector2, ScaleVector);
            FloatVector3 = _mm512_div_ps(FloatVector3, ScaleVector);

            FloatVector0 = _mm512_max_ps(FloatVector0, MinimumValueVector);
            FloatVector1 = _mm512_max_ps(FloatVector1, MinimumValueVector);
            FloatVector2 = _mm512_max_ps(FloatVector2, MinimumValueVector);
            FloatVector3 = _mm512_max_ps(FloatVector3, MinimumValueVector);

            FloatVector0 = _mm512_min_ps(FloatVector0, MaximumValueVector);
            FloatVector1 = _mm512_min_ps(FloatVector1, MaximumValueVector);
            FloatVector2 = _mm512_min_ps(FloatVector2, MaximumValueVector);
            FloatVector3 = _mm512_min_ps(FloatVector3, MaximumValueVector);

            auto IntegerVector0 = _mm512_cvtps_epi32(FloatVector0);
            auto IntegerVector1 = _mm512_cvtps_epi32(FloatVector1);
            auto IntegerVector2 = _mm512_cvtps_epi32(FloatVector2);
            auto IntegerVector3 = _mm512_cvtps_epi32(FloatVector3);

            IntegerVector0 = _mm512_add_epi32(IntegerVector0, ZeroPointVector);
            IntegerVector1 = _mm512_add_epi32(IntegerVector1, ZeroPointVector);
            IntegerVector2 = _mm512_add_epi32(IntegerVector2, ZeroPointVector);
            IntegerVector3 = _mm512_add_epi32(IntegerVector3, ZeroPointVector);

            MlasMaskedPackStore<typename OutputType>(D, __mmask16(_MM_K0_REG16), IntegerVector0);
            AccVector0 = _mm512_add_epi32(AccVector0, IntegerVector0);
            MlasMaskedPackStore<typename OutputType>(D + 16, __mmask16(_MM_K0_REG16), IntegerVector1);
            AccVector1 = _mm512_add_epi32(AccVector1, IntegerVector1);
            MlasMaskedPackStore<typename OutputType>(D + 32, __mmask16(_MM_K0_REG16), IntegerVector2);
            AccVector2 = _mm512_add_epi32(AccVector2, IntegerVector2);
            MlasMaskedPackStore<typename OutputType>(D + 48, __mmask16(_MM_K0_REG16), IntegerVector3);
            AccVector3 = _mm512_add_epi32(AccVector3, IntegerVector3);

            a += 64;
            D += 64;
            k -= 64;
        }

        // Reduce all accumulator vectors to AccVector0
        AccVector0 = _mm512_add_epi32(AccVector0, AccVector1);
        AccVector2 = _mm512_add_epi32(AccVector2, AccVector3);
        AccVector0 = _mm512_add_epi32(AccVector0, AccVector2);

        while (k >= 16) {
            auto FloatVector = _mm512_loadu_ps(a);
            FloatVector = _mm512_div_ps(FloatVector, ScaleVector);
            FloatVector = _mm512_max_ps(FloatVector, MinimumValueVector);
            FloatVector = _mm512_min_ps(FloatVector, MaximumValueVector);

            auto IntegerVector = _mm512_cvtps_epi32(FloatVector);
            IntegerVector = _mm512_add_epi32(IntegerVector, ZeroPointVector);

            MlasMaskedPackStore<typename OutputType>(D, __mmask16(_MM_K0_REG16), IntegerVector);
            AccVector0 = _mm512_add_epi32(IntegerVector, AccVector0);

            a += 16;
            D += 16;
            k -= 16;
        }

        if (k > 0) {
            __mmask16 mask = uint16_t((uint32_t(1) << k) - uint32_t(1));

            // output is padded with 0
            const auto AlignedN = k + (AlignedK - CountK);
            __mmask16 pad_mask = uint16_t((uint32_t(1) << AlignedN) - uint32_t(1));

            auto FloatVector = _mm512_maskz_loadu_ps(mask, a);
            FloatVector = _mm512_div_ps(FloatVector, ScaleVector);
            FloatVector = _mm512_max_ps(FloatVector, MinimumValueVector);
            FloatVector = _mm512_min_ps(FloatVector, MaximumValueVector);

            auto IntegerVector = _mm512_cvtps_epi32(FloatVector);
            IntegerVector = _mm512_add_epi32(IntegerVector, ZeroPointVector);

            MlasMaskedPackStore<typename OutputType>(D, pad_mask, IntegerVector);
            AccVector0 = _mm512_add_epi32(IntegerVector, AccVector0);
        }

        *RowSumBuffer++ = _mm512_reduce_add_epi32(AccVector0);

        A += lda;
        CountM -= 1;
    }
}

/**
 * @brief quantize and pack A to uint8_t
 */
void MLASCALL
MlasQuantizeLinearPackAKernelAvx512F(
    uint8_t* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    float Scale,
    uint8_t ZeroPoint
    )
{
    MlasQuantizeLinearPackAAvx512F<uint8_t>(D, A, lda, CountM, 
        CountK, RowSumBuffer, Scale, ZeroPoint);
}


/**
 * @brief quantize and pack A with 16b extension to int16_t
 */
void MLASCALL
MlasQuantizeLinearPackAExtKernelAvx512F(
    int16_t* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    float Scale,
    uint8_t ZeroPoint
    )
{
    MlasQuantizeLinearPackAAvx512F<int16_t>(D, A, lda, CountM, 
        CountK, RowSumBuffer, Scale, ZeroPoint);
}

