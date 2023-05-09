/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_q4.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for blocked int4 quantization and dequantization.

    Int4 block quantization is used to compress weight tensors of large
    language models.

--*/

#pragma once

#include "mlas.h"

#include <math.h>
#include <algorithm>

/**
 * @brief Computs the number of bytes required to pack and int4-quantize
 *        a weight matrix
 * @param N  the number of columns of matrix B. 
 * @param K  the number of rows of matrix B.
 * @return 
*/
size_t
MLASCALL
MlasQ4GemmPackBSize(
    size_t N,
    size_t K
    );

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
MlasQ4GemmPackB(
    void* PackedBuf,
    const float* FpData,
    size_t N,
    size_t K,
    size_t ldb
    );


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
MlasQ4GemmUnPackB(
    float* FpData,
    const void* PackedBuf,
    size_t N,
    size_t K,
    size_t ldb
    );
