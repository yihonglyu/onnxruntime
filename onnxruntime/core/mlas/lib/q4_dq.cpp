/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_q4.h

Abstract:

    This module contains the data structures and implementations
    for blocked int4 quantization and dequantization.

    Int4 block quantization is used to compress weight tensors of large
    language models.

--*/

#include "mlas_q4.h"

#include <math.h>
#include <algorithm>

/**
 * @brief # of fp32 numbers quantized together in block quantization.
 * Smaller number means more overhead (scale/zero-point storage) and
 * hopefully better accuracy.
 */
constexpr size_t MLAS_Q4_BLK_LEN = 32;

/*
 * The following functions describe an unaligned structure holding
 * a quantized block. The first 4 bytes forms a float32 scale.
 * Then we have a one byte zeropoint, only 4 bits are used. Next
 * are 16 bytes holding 32 4b numbers
 */

inline float&
MlasQ4BlkScale(uint8_t* BlkPtr)
{
    return *reinterpret_cast<float*>(BlkPtr);
}

inline float
MlasQ4BlkScale(const uint8_t* BlkPtr)
{
    return *reinterpret_cast<const float*>(BlkPtr);
}

inline uint8_t&
MlasQ4BlkZeroPoint(uint8_t* BlkPtr)
{
    return *(BlkPtr + sizeof(float));
}

inline uint8_t
MlasQ4BlkZeroPoint(const uint8_t* BlkPtr)
{
    return *(BlkPtr + sizeof(float));
}

inline uint8_t*
MlasQ4BlkData(uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float) + sizeof(uint8_t);
}

inline const uint8_t*
MlasQ4BlkData(const uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float) + sizeof(uint8_t);
}

constexpr size_t MLAS_Q4_BLK_SIZE = MLAS_Q4_BLK_LEN / 2 + sizeof(float) + sizeof(uint8_t);

//
// Quantization and Packing
//
// Since block quantization is used for compress large language model weights,
// it is usually used as the right hand side in matrix multiplications. So
// we can just perform quantize and packing together to help accelerate
// matrix multiplication.
//
// We take a tiles of 32 row and 4 column, transpose it, and quantize it
// into 4 blocks. So numbers in quantized block are from the same column.
// This is different from other int4 block quantization, where the numbers
// in a block are from the same row.
//

constexpr size_t MLAS_Q4_N_STRIDE = 4;

/**
 * @brief Computs the number of bytes required to pack and int4-quantize
 *        a weight matrix
 * @param N  the number of columns of matrix B.
 * @param K  the number of rows of matrix B.
 * @return
 */
size_t
MLASCALL
MlasQ4GemmPackBSize(size_t N, size_t K)
{
    const size_t AlignedN = (N + MLAS_Q4_N_STRIDE - 1) & ~(MLAS_Q4_N_STRIDE - 1);
    const size_t KBlocks = (K + MLAS_Q4_BLK_LEN - 1) / MLAS_Q4_BLK_LEN;

    const size_t NumBlocks = AlignedN * KBlocks;

    return NumBlocks * MLAS_Q4_BLK_SIZE;
}

/**
 * @brief Prepack and Quantize fp32 weight tensor to int4 blocks
 * @param PackedBuf  destination buffer
 * @param FpData     the pointer to fp32 matrix
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
 */
void
MLASCALL
MlasQ4GemmPackB(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb)
{
    auto* dst_ptr = reinterpret_cast<uint8_t*>(PackedBuf);

    for (size_t n = 0; n < N; n += MLAS_Q4_N_STRIDE) {
        size_t nlen = std::min(MLAS_Q4_N_STRIDE, N - n);

        for (size_t k = 0; k < K; k += MLAS_Q4_BLK_LEN) {
            size_t klen = std::min(MLAS_Q4_BLK_LEN, K - k);

            const float* src = FpData + ldb * k + n;

            for (size_t nn = 0; nn < nlen; nn++) {
                float min = std::numeric_limits<float>::max();
                float max = -min;

                for (size_t l = 0; l < klen; l++) {
                    const float v = src[ldb * l];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
                min = std::min(min, 0.0f);
                max = std::max(max, 0.0f);

                const float scale = (max - min) / ((1 << 4) - 1);
                const float reciprocal_scale = scale ? 1.0f / scale : 0.0f;
                float zero_point_fp = min;
                if (scale != 0.0f) {
                    zero_point_fp = 0.f - min / scale;
                }

                // Handle any clamping
                uint8_t& zp = MlasQ4BlkZeroPoint(dst_ptr);
                if (zero_point_fp < 0.0f) {
                    zp = 0;
                } else if (zero_point_fp > 15.0f) {
                    zp = 15;
                } else {
                    zp = (uint8_t)roundf(zero_point_fp);
                }
                MlasQ4BlkScale(dst_ptr) = scale;
                uint8_t* data = MlasQ4BlkData(dst_ptr);

                for (size_t l = 0; l < MLAS_Q4_BLK_LEN; l += 2) {
                    const float v0 = l < klen ? src[ldb * l] : 0;
                    const uint8_t vi0 = (uint8_t)std::min(
                        15.0f, std::max(0.0f, roundf(v0 * reciprocal_scale + zp)));
                    const float v1 = (l + 1 < klen) ? src[ldb * (l + 1)] : 0;
                    const uint8_t vi1 = (uint8_t)std::min(
                        15.0f, std::max(0.0f, roundf(v1 * reciprocal_scale + zp)));

                    data[l / 2] = vi0 | (vi1 << 4);
                }
                dst_ptr += MLAS_Q4_BLK_SIZE;
                src++;  // mov to next column
            }
            if (nlen < MLAS_Q4_N_STRIDE) {
                memset(dst_ptr, 0, MLAS_Q4_BLK_SIZE * (MLAS_Q4_N_STRIDE - nlen));
                dst_ptr += MLAS_Q4_BLK_SIZE * (MLAS_Q4_N_STRIDE - nlen);
            }

        }  // advance to next block or rows
    }      // advance next block of columns
}

/**
 * @brief Unpack and dequantize from int4 to fp32, reverse operation of
 *        MlasQ4GemmPackB
 * @param FpData     destination buffer, the fp32 matrix
 * @param PackedBuf  int4 quantized and packed data
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
 */
void
MLASCALL
MlasQ4GemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb)
{
    const auto* src = reinterpret_cast<const uint8_t*>(PackedBuf);
    for (size_t n = 0; n < N; n += MLAS_Q4_N_STRIDE) {
        size_t CountN = std::min(N - n, MLAS_Q4_N_STRIDE);

        for (size_t k = 0; k < K; k += MLAS_Q4_BLK_LEN) {
            size_t CountK = std::min(K - k, MLAS_Q4_BLK_LEN);

            float* dest = FpData + ldb * k + n;
            for (size_t nn = 0; nn < CountN; nn++) {
                const float s = MlasQ4BlkScale(src);
                const uint8_t z = MlasQ4BlkZeroPoint(src);
                const uint8_t* pp = MlasQ4BlkData(src);

                for (size_t l = 0; l < CountK; l += 2) {
                    const uint8_t vi = pp[l / 2];

                    const int8_t vi0 = vi & 0x0F;
                    const int8_t vi1 = vi >> 4;

                    const float v0 = (vi0 - z) * s;
                    const float v1 = (vi1 - z) * s;

                    dest[ldb * l] = v0;
                    if (l + 1 < CountK) {
                        dest[ldb * (l + 1)] = v1;
                    }
                }
                src += MLAS_Q4_BLK_SIZE;
                dest++;  // next column
            }
            src += (MLAS_Q4_N_STRIDE - CountN) * MLAS_Q4_BLK_SIZE;
        }
    }
}
