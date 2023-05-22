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
 * @brief Define types of block quantization
 */
typedef enum {
    BlkQ4Sym = 0,  /*!< int4 Symmetric Block Quantization, zero_point = 0 */
    BlkQ4Zp8 = 1   /*!< int4 Block Quantization, zero_point is int8 type */
}MLAS_BLK_QUANT_TYPE;

/**
 * @brief Computs the number of bytes required to pack and int4-quantize
 *        a weight matrix
 * @param QType  type of block quantization
 * @param N      the number of columns of matrix B. 
 * @param K      the number of rows of matrix B.
 * @return 
*/
size_t
MLASCALL
MlasQ4GemmPackBSize(
    MLAS_BLK_QUANT_TYPE QType,
    size_t N,
    size_t K
    );

/**
 * @brief Prepack and Quantize fp32 weight tensor to int4 blocks
 * 
 * @param QType      type of block quantization
 * @param PackedBuf  destination buffer
 * @param FpData     the pointer to fp32 matrix
 * @param N          the number of columns of matrix B. 
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
*/
void
MLASCALL
MlasQ4GemmPackB(
    MLAS_BLK_QUANT_TYPE QType,
    void* PackedBuf,
    const float* FpData,
    size_t N,
    size_t K,
    size_t ldb
    );


/**
 * @brief Unpack and dequantize from int4 to fp32, reverse operation of
 *        MlasQ4GemmPackB
 * @param QType      type of block quantization
 * @param FpData     destination buffer, the fp32 matrix
 * @param PackedBuf  int4 quantized and packed data
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
 */
void
MLASCALL
MlasQ4GemmUnPackB(
    MLAS_BLK_QUANT_TYPE QType,
    float* FpData,
    const void* PackedBuf,
    size_t N,
    size_t K,
    size_t ldb
    );


template<typename T>
class MLAS_GEMM_POSTPROCESSOR
{
   public:
    virtual void Process(T*,         /**< the address of matrix to process */
                         size_t,     /**< the start row index of matrix */
                         size_t,     /**< the start col index of matrix */
                         size_t,     /**< the element count per row to process */
                         size_t,     /**< the element count per col to process */
                         size_t      /**< the leading dimension of matrix */
    ) const = 0;

    virtual ~MLAS_GEMM_POSTPROCESSOR() {}
};


/**
 * @brief Data parameters for Q4 GEMM routine
 *        C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed int4 blob
 *        All except C are [in] parameters
 */
struct MLAS_Q4_GEMM_DATA_PARAMS {
    const float* A = nullptr;        /**< address of A (float32 matrix)*/
    const void* B = nullptr;         /**< address of B (quantized and packed int4 blob)*/
    const float* Bias = nullptr;     /**< address of Bias, vector size N */
    float* C = nullptr;              /**< address of result matrix */
    size_t lda = 0;                  /**< leading dimension of A */
    size_t ldc = 0;                  /**< leading dimension of C*/
    const MLAS_GEMM_POSTPROCESSOR<float>* OutputProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed int4 blob
 *
 * @param[in]  QType   type of block quantization used in B
 * @param[in]  M       row size of matrix A and C
 * @param[in]  N       column size of matrix B and C
 * @param[in]  K       column size of matrix A and row size of matrix B
 * @param[in]  BatchN  number of batches
 * @param[inout]  DataParams  An array (size BatchN) of parameter blocks
 * @param[in]  ThreadPool
 * @return
 */
void MLASCALL
MlasQ4GemmBatch(
    MLAS_BLK_QUANT_TYPE QType,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool = nullptr
    );
