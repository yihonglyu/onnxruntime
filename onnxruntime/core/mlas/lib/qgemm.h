/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm.h

Abstract:

    This module defines the set of template functions to implement a kernel of
    quantized integer matrix/matrix multiply operation (QGEMM).

    To implement a new kernel, there needs to specialize template functions below:
        MlasGemmU8X8FixupZeroPointA
        MlasGemmU8X8FixupZeroPointB
        MlasGemmU8X8CopyPackA
        MlasGemmU8X8CopyPackB
        MlasGemmU8X8Kernel
    Specialization of MlasGemmU8X8TryGemvKernel is optional.

    MlasGemmU8X8Operation and MlasGemmU8X8PackedOperation are shared kernel drivers.
    MlasGemmU8X8ScaleSumBuffer is a helper function.

    It also includes the dispatcher logics.

--*/

#pragma once

#include "mlasi.h"

//
// Define the parameters to execute segments of a QGEMM operation on worker
// threads.
//

struct MLAS_GEMM_U8X8_WORK_BLOCK {
    void* PackedA;
    void* PackedB;
    bool SingleThreaded;
};

//
// Define the default striding parameters used for the quantized integer
// matrix/matrix multiply operation.
//

struct MLAS_GEMM_U8X8_STRIDES {
    size_t M;
    size_t N;
    size_t K;
};

template<typename KernelType>
MLAS_FORCEINLINE
bool
MlasGemmU8X8TryGemvKernel(
    const uint8_t* A,
    const uint8_t* B,
    size_t ldb,
    int32_t* C,
    size_t CountK,
    size_t CountN,
    bool BIsSigned
)
{
    MLAS_UNREFERENCED_PARAMETER(A);
    MLAS_UNREFERENCED_PARAMETER(B);
    MLAS_UNREFERENCED_PARAMETER(ldb);
    MLAS_UNREFERENCED_PARAMETER(C);
    MLAS_UNREFERENCED_PARAMETER(CountK);
    MLAS_UNREFERENCED_PARAMETER(CountN);
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    return false;
}

template <typename KernelType>
MLAS_FORCEINLINE int32_t
MlasGemmU8X8FixupZeroPointA(int32_t ZeroPointA)
{
    return ZeroPointA;
}

template<typename KernelType>
int32_t
MlasGemmU8X8FixupZeroPointB(
    int32_t ZeroPointB,
    bool BIsSigned
)
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    return ZeroPointB;
}

template<typename KernelType>
MLAS_FORCEINLINE
void
MlasGemmU8X8FixupZeroPointB(
    const uint8_t* PackedZeroPointB,
    int32_t* ZeroPointBBuffer,
    size_t N,
    bool BIsSigned
)
{
    int32_t ZeroPointB;

    for (size_t n = 0; n < N; n++) {

        ZeroPointB = typename KernelType::OffsetBType(PackedZeroPointB[n]);
        ZeroPointB = MlasGemmU8X8FixupZeroPointB<KernelType>(ZeroPointB, BIsSigned);

        ZeroPointBBuffer[n] = -ZeroPointB;
    }

    //
    // Fill the misaligned slots of the zero point buffer with zeroes to guard
    // against tools that check for uninitialized data usage.
    //

    size_t AlignedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);

    for (size_t n = N; n < AlignedN; n++) {
        ZeroPointBBuffer[n] = 0;
    }
}

template<typename KernelType>
void
MlasGemmU8X8CopyPackA(
    typename KernelType::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
);

template<typename KernelType>
void
MlasGemmU8X8CopyPackB(
    typename KernelType::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
);

template<typename KernelType>
size_t
MlasGemmU8X8Kernel(
    const typename KernelType::PackedAType* A,
    const typename KernelType::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
);


inline
void
MlasGemmU8X8ScaleSumBuffer(
    int32_t* Output,
    const int32_t* Input,
    size_t N,
    int32_t Scale
)
{
    for (size_t n = 0; n < N; n++) {
        Output[n] = Input[n] * Scale;
    }
}


MLAS_FORCEINLINE
void
MlasGemmU8X8ScaleSumBuffer(
    int32_t* SumBuffer,
    size_t N,
    int32_t Scale
)
{
    return MlasGemmU8X8ScaleSumBuffer(SumBuffer, SumBuffer, N, Scale);
}

typedef size_t(MLAS_GEMM_U8X8_PACKA_SIZE_ROUTINE)(size_t M, size_t K);
typedef size_t(MLAS_GEMM_U8X8_PACKB_SIZE_ROUTINE)(size_t N, size_t K);

constexpr uint64_t PackStatusNotStarted = 0x7FFFFFFF7FFFFFFF;
constexpr uint64_t PackStatusInProgress = 0x7FFFFFFE7FFFFFFE;

/**
 * @brief Compute the size of the packed A buffer needed
 *
 * @tparam KernelType
 * @param M   Number of rows in A
 * @param K   Number of columns in A
 * @return    The packed A buffer size in bytes
 */
template <typename KernelType>
size_t
MlasGemmPackASizeT(size_t M, size_t K)
{
    using PackedT = typename KernelType::PackedAType;

    //
    // compute the row sum space needed
    PackedT* BufStartPtr = reinterpret_cast<PackedT*>(0);
    int32_t* RowSums;
    PackedT* PackedPtr;
    MlasGetSumValBufFromPackedMutable<PackedT>(BufStartPtr, M, RowSums, PackedPtr);
    size_t RowSumSize = reinterpret_cast<size_t>(PackedPtr) - reinterpret_cast<size_t>(BufStartPtr);

    //
    // compute the packed matrix size
    constexpr size_t PK = KernelType::PackedK;
    const size_t AlignedK = (K + PK - 1) & ~(PK - 1);
    size_t PackedDataSize = M * AlignedK * sizeof(PackedT);

    size_t BytesRequired = RowSumSize + PackedDataSize;
    BytesRequired += 16;  // tolerate a buffer overrun bug in neon pack a
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired =
        (BytesRequired + BufferAlignment - 1) & ~(BufferAlignment - 1);
    return AlignedBytesRequired;
}

/**
 * @brief Compute the size of the packed B buffer needed
 *
 * @tparam KernelType
 * @param N          Number of columns in B
 * @param K          Number of columns in A
 * @param BIsSigned  Whether packed B is int8_t or uint8_t
 * @return           The packed B buffer size in bytes
 */
template <typename KernelType>
size_t
MlasGemmPackBSizeT(size_t N, size_t K)
{
    using PackedT = typename KernelType::PackedBType;

    //
    // compute the row sum space needed
    PackedT* BufStartPtr = reinterpret_cast<uint8_t*>(0);
    int32_t* ColSums;
    PackedT* PackedPtr;
    MlasGetSumValBufFromPackedMutable<PackedT>(BufStartPtr, N, ColSums, PackedPtr);
    size_t HeaderSize = reinterpret_cast<size_t>(PackedPtr) - reinterpret_cast<size_t>(BufStartPtr);

    constexpr size_t PK = KernelType::PackedK;

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const size_t AlignedK = (K + PK - 1) & ~(PK - 1);

    const size_t BytesRequired = HeaderSize + (AlignedN * AlignedK * sizeof(PackedT));
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired =
        (BytesRequired + BufferAlignment - 1) & ~(BufferAlignment - 1);

    return AlignedBytesRequired;
}

/**
 * @brief A single threaded job for pre-packing B
 * @tparam KernelType 
 * @param N            Supplies the number of columns of matrix B.
 * @param K            Supplies the the number of rows of matrix B.
 * @param RangeStartN  Starting column of this job
 * @param RangeCountN  Number of columns for this job
 * @param B            Supplies the address of matrix B.
 * @param ldb          Supplies the first dimension of matrix B.
 * @param BIsSigned    Supplies true if matrix B is signed data, else
                       false if matrix B is unsigned data.
 * @param PackedBBuf   Supplies the address of packed matrix B.
*/
template<typename KernelType>
void
MlasGemmU8X8CopyPackBThreaded(
    size_t N,
    size_t K,
    size_t RangeStartN,
    size_t RangeCountN,
    const uint8_t* B,
    size_t ldb,
    bool BIsSigned,
    void* PackedBBuf
    )
{
    //
    // Retrieve the packing parameters.
    //
    constexpr size_t PackedK = KernelType::PackedK;

    //
    // Reserve and initialize storage for the column sum buffer to hold the sums
    // of the elements along each of the columns.
    //
    int32_t* PackedColumnSumBuffer;
    typename KernelType::PackedBType* PackedB;
    MlasGetSumValBufFromPackedMutable(PackedBBuf, N, PackedColumnSumBuffer, PackedB);

    // First couple of column buffers are reused for packing status.
    // We leave the first cache line untouched until the last moment
    MLAS_DECLSPEC_ALIGN(int32_t FirstLine[16], 64);
    std::fill_n(FirstLine, 16, 0);

    PackedColumnSumBuffer += RangeStartN;
    if (RangeCountN > 16) {
        std::fill_n(PackedColumnSumBuffer + 16, RangeCountN - 16, 0);
    }

    B += RangeStartN;

        //
        // Step through each slice of matrix B along the N dimension.
        //

        const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);
        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {
            typename KernelType::PackedBType* pb = PackedB + (RangeStartN + n) * AlignedK;
            constexpr size_t BatchedN = 128;
            MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[BatchedN], 64);

            CountN = std::min(RangeCountN - n, BatchedN);

            MlasGemmU8X8CopyPackB<KernelType>(pb, B + n, ldb, CountN, K, ColumnSumBuffer,
                                               BIsSigned);

            //
            // Accumulate this batch of the column sum buffer into the packed
            // buffer accumulators.
            //

            for (size_t nn = 0; nn < CountN; nn++) {
                if (n + nn < 16) {
                    FirstLine[n + nn] += ColumnSumBuffer[nn];
                } else {
                    PackedColumnSumBuffer[n + nn] += ColumnSumBuffer[nn];
                }
            }
        }

    // Column buffers are padded to cacheline align, so a little
    // overrun here is ok
    memcpy(PackedColumnSumBuffer + 2, FirstLine + 2, 56);
    auto* StoreDst = reinterpret_cast<std::atomic<uint64_t>*>(PackedColumnSumBuffer);
    StoreDst->store(*((uint64_t*)FirstLine), std::memory_order_release);
}


template <typename KernelType>
bool
TryPackA(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    size_t RangeStartM,
    size_t RangeCountM
   )
{

    RangeCountM = std::min(RangeCountM, Shape->M - RangeStartM);

    int32_t* PackedRowSumBuffer;
    typename KernelType::PackedAType* PackedA;
    MlasGetSumValBufFromPackedMutable<typename KernelType::PackedAType>(
        WorkBlock->PackedA, Shape->M, PackedRowSumBuffer, PackedA);
    PackedRowSumBuffer += RangeStartM;

    if (WorkBlock->SingleThreaded) {

        uint64_t& status = *(uint64_t*)(PackedRowSumBuffer);
        if (status == PackStatusNotStarted) {

            auto A = Data->A + RangeStartM * Data->lda;
            const size_t AlignedK = (Shape->K + KernelType::PackedK - 1) & ~(KernelType::PackedK - 1);
            PackedA += RangeStartM * AlignedK;

            MlasGemmU8X8CopyPackA<KernelType>(PackedA, A, Data->lda, RangeCountM, Shape->K,
                                              PackedRowSumBuffer);
        }
        return true;
    }

    std::atomic<uint64_t>& status = *((std::atomic<uint64_t>*)(PackedRowSumBuffer));
    uint64_t StatusVal = status.load(std::memory_order_relaxed);

    if (StatusVal == PackStatusNotStarted) {
        uint64_t exchanged_status = PackStatusNotStarted;
        if (status.compare_exchange_strong(exchanged_status, PackStatusInProgress,
                                           std::memory_order_acq_rel, std::memory_order_acquire)) {
            // So we successfuly changed the status from not started to in progress.
            // Now pack A

            // First couple of row sums are reused for packing status.
            // We leave the first cache line untouched until the last moment
            constexpr size_t HeadBufSize = 64;
            MLAS_DECLSPEC_ALIGN(int32_t FirstLine[HeadBufSize], 64);

            auto A = Data->A + RangeStartM * Data->lda;
            const size_t AlignedK =
                (Shape->K + KernelType::PackedK - 1) & ~(KernelType::PackedK - 1);
            PackedA += RangeStartM * AlignedK;

            const size_t FirstBatch = std::min(RangeCountM, HeadBufSize);
            MlasGemmU8X8CopyPackA<KernelType>(PackedA, A, Data->lda, FirstBatch, Shape->K,
                                              FirstLine);
            if (RangeCountM > HeadBufSize) {
                MlasGemmU8X8CopyPackA<KernelType>(PackedA + HeadBufSize * AlignedK, A + HeadBufSize * Data->lda,
                                                  Data->lda, RangeCountM - HeadBufSize, Shape->K,
                                                  PackedRowSumBuffer + HeadBufSize);
            }
            if (FirstBatch > 2) {
                memcpy(PackedRowSumBuffer + 2, FirstLine + 2, (FirstBatch - 2)*sizeof(int32_t));
            }
            auto* StoreDst = reinterpret_cast<std::atomic<uint64_t>*>(PackedRowSumBuffer);
            StoreDst->store(*((uint64_t*)FirstLine), std::memory_order_release);

            return true;
        } else if (exchanged_status == PackStatusInProgress) {
            // Another thread is currently packing this block.
            return false;
        } else {
            // Another thread already packed this block.
            return true;
        }
    }

    if (StatusVal == PackStatusInProgress) {
        if (status.load(std::memory_order_acquire) == PackStatusInProgress) {
            return false;
        }
    }

    // Another thread already packed this block.
    return true;
}

template <typename  KernelType>
bool
TryPackB(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    size_t RangeStartN,
    size_t RangeCountN)
{
    if (Data->BIsPacked) {
        return true;
    }

    RangeCountN = std::min(RangeCountN, Shape->N - RangeStartN);

    int32_t* PackedColSumBuffer;
    typename KernelType::PackedBType* PackedB;
    MlasGetSumValBufFromPackedMutable<typename KernelType::PackedBType>(
        WorkBlock->PackedB, Shape->N, PackedColSumBuffer, PackedB);
    PackedColSumBuffer += RangeStartN;

    if (WorkBlock->SingleThreaded) {
        uint64_t& status = *(uint64_t*)(PackedColSumBuffer);
        if (status == PackStatusNotStarted) {
            auto B = (uint8_t*)(Data->B) + RangeStartN;
            const size_t AlignedK =
                (Shape->K + KernelType::PackedK - 1) & ~(KernelType::PackedK - 1);
            PackedB += RangeStartN * AlignedK;

            MlasGemmU8X8CopyPackB<KernelType>(PackedB, B, Data->ldb, RangeCountN, Shape->K,
                                              PackedColSumBuffer, Shape->BIsSigned);

        }
        return true;
    }

    std::atomic<uint64_t>& status = *((std::atomic<uint64_t>*)(PackedColSumBuffer));
    uint64_t StatusVal = status.load(std::memory_order_relaxed);
    if (StatusVal == PackStatusNotStarted) {
        uint64_t exchanged_status = PackStatusNotStarted;
        if (status.compare_exchange_strong(exchanged_status, PackStatusInProgress,
                                           std::memory_order_acq_rel, std::memory_order_acquire)) {
            // So we successfuly changed the status from not started to in progress.

            // First couple of row sums are reused for packing status.
            // We leave the first cache line untouched until the last moment
            MLAS_DECLSPEC_ALIGN(int32_t FirstLine[16], 64);

            auto B = (uint8_t*)(Data->B) + RangeStartN;
            const size_t AlignedK =
                (Shape->K + KernelType::PackedK - 1) & ~(KernelType::PackedK - 1);
            PackedB += RangeStartN * AlignedK;

            if (RangeCountN > 16) {
                MlasGemmU8X8CopyPackB<KernelType>(PackedB + 16 * AlignedK, B + 16,
                                                  Data->ldb, RangeCountN - 16, Shape->K,
                                                  PackedColSumBuffer + 16, Shape->BIsSigned);
            }
            MlasGemmU8X8CopyPackB<KernelType>(PackedB, B, Data->ldb,
                                              std::min(RangeCountN, (size_t)16), Shape->K,
                                              FirstLine, Shape->BIsSigned);

            // Row buffers are padded to cacheline align, so a little
            // overrun here is ok
            memcpy(PackedColSumBuffer + 2, FirstLine + 2, 56);
            auto* StoreDst = reinterpret_cast<std::atomic<uint64_t>*>(PackedColSumBuffer);
            StoreDst->store(*((uint64_t*)FirstLine), std::memory_order_release);

            return true;
        } else if (exchanged_status == PackStatusInProgress) {
            // Another thread is currently packing this block.
            return false;
        } else {
            // Another thread already packed this block.
            return true;
        }
    }

    if (StatusVal == PackStatusInProgress) {
        if (status.load(std::memory_order_acquire) == PackStatusInProgress) {
            return false;
        }
    }

    // Another thread already packed this block.
    return true;
}


template <typename KernelType>
void
EnsurePacked(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    size_t RangeStartM,
    size_t RangeCountM,
    size_t RangeStartN,
    size_t RangeCountN
    )
{
    bool Apacked = false;
    bool Bpacked = false;

    // variables for look ahead packing
    //size_t nextM = RangeStartM;
    //size_t nextN = RangeStartN;
    //bool nextSideA = false;

    while (true) {
        if (!Apacked) {
            Apacked = TryPackA<KernelType>(Shape, Data, WorkBlock, RangeStartM, RangeCountM);
        }
        if (!Bpacked) {
            Bpacked = TryPackB<KernelType>(Shape, Data, WorkBlock, RangeStartN, RangeCountN);
        }
        if (Apacked && Bpacked) {
            // mission acomplished.
            break;
        }

        // someone else is packing for us. trying to do some work for others while
        // waiting
        //nextSideA = !nextSideA;
        //if (nextSideA) {
        //    nextM += RangeCountM;
        //    if (nextM >= Shape->M) {
        //        nextM = 0;
        //    }
        //    if (nextM != RangeStartM) {
        //        TryPackA<KernelType>(Shape, Data, nextM, RangeCountM);
        //    }
        //} else {
        //    nextN += RangeCountN;
        //    if (nextN >= Shape->N) {
        //        nextN = 0;
        //    }
        //    if (nextN != RangeStartN) {
        //        TryPackB<KernelType>(Shape, Data, nextN, RangeCountN);
        //    }
        //}
    }
}

template <typename KernelType>
void
MlasGemmU8X8Operation(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    size_t RangeStartM,
    size_t RangeCountM,
    size_t RangeStartN,
    size_t RangeCountN
    )
{
    EnsurePacked<KernelType>(Shape, Data, WorkBlock, RangeStartM, RangeCountM, RangeStartN, RangeCountN);

    // These counts are actually strides shared by all threaded tasks.
    RangeCountM = std::min(RangeCountM, Shape->M - RangeStartM);
    RangeCountN = std::min(RangeCountN, Shape->N - RangeStartN);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumBuffer[1024], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[1024], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ZeroPointBBuffer[1024], 64);

    const size_t K = Shape->K;

    const size_t ldc = Data->ldc;

    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB =
        Data->PerColumnZeroPoints ? Data->ZeroPointB + RangeStartN : nullptr;

    int32_t ZeroPointA = Data->ZeroPointA;
    int32_t ZeroPointB = typename KernelType::OffsetBType(*Data->ZeroPointB);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix A if the
    // kernel requires signed data.
    //

    ZeroPointA = MlasGemmU8X8FixupZeroPointA<KernelType>(ZeroPointA);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix B if the
    // data is the opposite format of the kernel implementation. This value is
    // ignored if per-column zero point offsets are used instead.
    //

    ZeroPointB = MlasGemmU8X8FixupZeroPointB<KernelType>(ZeroPointB, Shape->BIsSigned);

    //
    // Extract the pointer to the row/column sum buffer from the packed matrix.
    //

    const int32_t* PackedColumnSumBuffer = nullptr;
    const typename KernelType::PackedBType* PackedB = nullptr;
    const void* SrcPackedB = Data->BIsPacked ? Data->B : WorkBlock->PackedB;
    MlasGetSumValBufFromPacked(SrcPackedB, Shape->N, PackedColumnSumBuffer, PackedB);
    PackedColumnSumBuffer += RangeStartN;

    const int32_t* PackedRowSumBuffer = nullptr;
    const typename KernelType::PackedAType* PackedA = nullptr;
    MlasGetSumValBufFromPacked(WorkBlock->PackedA, Shape->M, PackedRowSumBuffer, PackedA);
    PackedRowSumBuffer += RangeStartM;


        const size_t PackedCountK = (K + KernelType::PackedK - 1) / KernelType::PackedK;

        //
        // Step through each slice of matrix B along the N dimension.
        //
        {
            MlasGemmU8X8ScaleSumBuffer(ColumnSumBuffer, PackedColumnSumBuffer, RangeCountN,
                                           -ZeroPointA);

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmU8X8FixupZeroPointB<KernelType>(PackedZeroPointB, ZeroPointBBuffer,
                                                        RangeCountN, Shape->BIsSigned);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const auto* b = PackedB + RangeStartN * KernelType::PackedK * PackedCountK;
            int32_t* c = C;

            {
                const typename KernelType::PackedAType* pa =
                    PackedA + RangeStartM * KernelType::PackedK * PackedCountK;

                    //
                    // Apply the global depth value constant without the ZeroPointB scaling from:
                    //
                    //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                    //              ==>
                    //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA *
                    //     ZeroPointB
                    //
                    // The ZeroPointB term is factored out and either applied below for per-matrix
                    // quantization or inside the kernel for per-column quantization.
                    //
                    // SUM(A[i] * B[i] - B[i] * ZeroPointA - (A[i] - ZeroPointA) * ZeroPointB)
                    // SUM(A[i] * B[i]) - SUM(B[i]) * ZeroPointA - (SUM(A[i]) - K * ZeroPointA)
                    // * ZeroPointB

                    for (size_t mm = 0; mm < RangeCountM; mm++) {
                        RowSumBuffer[mm] = PackedRowSumBuffer[mm] - int32_t(K) * ZeroPointA;
                    }

                    //
                    // Scale the row sums by the per-matrix zero point offset of matrix B.
                    //

                    if (PackedZeroPointB == nullptr) {
                        MlasGemmU8X8ScaleSumBuffer(RowSumBuffer, RangeCountM, -ZeroPointB);
                    }
                //
                // Step through the rows of the local packed buffer.
                //

                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = RangeCountM;

                bool ZeroMode = true;
                bool PostProcess = true;

                while (RowsRemaining > 0) {
                    size_t RowsHandled = MlasGemmU8X8Kernel<KernelType>(
                        pa, b, c, PackedCountK, RowsRemaining, RangeCountN, ldc, RowSums,
                        ColumnSumBuffer, (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr,
                        ZeroMode);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                            RowsHandled, RangeCountN, Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

}


//
// Quantized integer matrix/matrix dispatch structure.
//

typedef
void
(MLAS_GEMM_U8X8_OPERATION)(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    );

typedef
void
(MLAS_GEMM_U8X8_COPY_PACKB_ROUTINE)(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    );

struct MLAS_GEMM_U8X8_DISPATCH {
    MLAS_GEMM_U8X8_OPERATION* Operation;
    MLAS_GEMM_U8X8_COPY_PACKB_ROUTINE* CopyPackBRoutine;
    size_t PackedK;
    size_t PackedStrideK;
    MLAS_GEMM_U8X8_PACKA_SIZE_ROUTINE* PackASize;
    MLAS_GEMM_U8X8_PACKB_SIZE_ROUTINE* PackBSize;
};

#define USE_NEONS8_KERNEL true

MLAS_FORCEINLINE
const MLAS_GEMM_U8X8_DISPATCH*
MlasGemmU8X8GetDispatch(
    bool BIsSigned
)
{
    const MLAS_GEMM_U8X8_DISPATCH* GemmU8X8Dispatch;

    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

#if defined(MLAS_TARGET_AMD64_IX86)
    if (BIsSigned) {
        GemmU8X8Dispatch = MlasPlatform.GemmU8S8Dispatch;
    }
    else {
        GemmU8X8Dispatch = MlasPlatform.GemmU8U8Dispatch;
    }
#elif defined(MLAS_TARGET_ARM64)
    GemmU8X8Dispatch = MlasPlatform.GemmU8X8Dispatch;
    if (USE_NEONS8_KERNEL && BIsSigned && GemmU8X8Dispatch == &MlasGemmU8X8DispatchNeon) {
        GemmU8X8Dispatch = &MlasGemmS8S8DispatchNeon;
    }
#elif defined(MLAS_TARGET_ARM64EC) || (defined(MLAS_TARGET_ARM) && !defined(_MSC_VER))
    GemmU8X8Dispatch = &MlasGemmU8X8DispatchNeon;
#else
    GemmU8X8Dispatch = &MlasGemmU8X8DispatchDefault;
#endif

    return GemmU8X8Dispatch;
}
