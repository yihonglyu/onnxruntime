/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm.cpp

Abstract:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

--*/
#include <chrono>
#include "mlasi.h"
#include "qgemm.h"

void
MLASCALL
MlasGemm(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS &Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS &DataParams,
    MLAS_THREADPOOL *ThreadPool)
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    MlasGemmBatch(Shape, &DataParams, 1, ThreadPool);
}

thread_local size_t QGemmId;
thread_local size_t QGemmPackM;

void
MLASCALL
MlasGemmBatch(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool)
{
    const size_t M = Shape.M;
    const size_t N = Shape.N;
    const size_t K = Shape.K;

    const auto* GemmU8X8Dispatch = MlasGemmU8X8GetDispatch(Shape.BIsSigned);
    MLAS_GEMM_U8X8_OPERATION* GemmU8X8Operation = GemmU8X8Dispatch->Operation;
    MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlocks;

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K);
    ptrdiff_t ThreadsPerGemm = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    const size_t PackASize = GemmU8X8Dispatch->PackASize(M, K);
    std::unique_ptr<uint8_t[]> PackABufPtr =
        std::make_unique<uint8_t[]>((PackASize + sizeof(MLAS_GEMM_U8X8_WORK_BLOCK)) * BatchN + 64);
    ptrdiff_t addr = (ptrdiff_t)PackABufPtr.get();
    WorkBlocks = (MLAS_GEMM_U8X8_WORK_BLOCK*)addr;
    addr += BatchN * sizeof(MLAS_GEMM_U8X8_WORK_BLOCK);

    addr = (addr + 63) & ~63;
    for (size_t b = 0; b < BatchN; b++) {
        WorkBlocks[b].PackedA = (void*)addr;
        addr += PackASize;
        WorkBlocks[b].PackedB = nullptr;
        WorkBlocks[b].SingleThreaded = ThreadsPerGemm == 1 || MlasGetMaximumThreadCount(ThreadPool) <= 1;
    }

    std::unique_ptr<uint8_t[]> PackBBufPtr;
    if (!DataParams->BIsPacked) {
        // TODO!! mixed b packing???
        const size_t PackBSize = GemmU8X8Dispatch->PackBSize(N, K);
        PackBBufPtr = std::make_unique<uint8_t[]>(PackBSize * BatchN + 64);
        addr = (ptrdiff_t)PackBBufPtr.get();
        addr = (addr + 63) & ~63;
        for (size_t b = 0; b < BatchN; b++) {
            WorkBlocks[b].PackedB = (void*)addr;
            addr += PackBSize;
        }
    }

     size_t StrideM = M;
     size_t StrideN = N;
     size_t ThreadCountM = 1;
     size_t ThreadCountN = 1;

     if (ThreadsPerGemm > 1) {
        //
        // Segment the operation across multiple threads.
        //

        if ((N * ThreadsPerGemm) <= (M + M / 8)) {
            //
            // tall and thin, one dimensional partition
            //

            const ptrdiff_t BlockedM =
                (M + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) / MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
            StrideM = ((BlockedM + ThreadsPerGemm - 1) / ThreadsPerGemm) *
                      MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
        } else if ((M * ThreadsPerGemm) <= (N + N / 8)) {
            //
            // short and flat, one  dimensional partition
            //

            const ptrdiff_t BlockedN =
                (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) / MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
            StrideN = ((BlockedN + ThreadsPerGemm - 1) / ThreadsPerGemm) *
                                MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
        } else {
            //
            // Try to partition the resulting matrix into shapes such that N is about 2*M,
            //

            const double sq_area = double(M) * double(N) / ThreadsPerGemm;
            const double sq_edge = sqrt(sq_area * 2);

            StrideN = size_t(ceil(sq_edge / MLAS_QGEMM_STRIDEN_THREAD_ALIGN)) *
                                MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

            StrideM = (size_t)ceil(sq_area / (StrideN * MLAS_QGEMM_STRIDEN_THREAD_ALIGN)) *
                      MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
        }
        ThreadCountM = MlasDivRndup(M, StrideM);
        ThreadCountN = MlasDivRndup(N, StrideN);
        ThreadsPerGemm = ThreadCountM * ThreadCountN;
     }

    //  zero out packing status 
    for (size_t batchIdx = 0; batchIdx < BatchN; batchIdx++) {
        auto& data = WorkBlocks[batchIdx];
        int32_t* SumBuffer;
        uint8_t* Packed;

            MlasGetSumValBufFromPackedMutable<uint8_t>(data.PackedA, M, SumBuffer, Packed);
            for (size_t m = 0; m < M; m += StrideM) {
                uint64_t* PackStatus = (uint64_t*)(SumBuffer + m);
                *PackStatus = PackStatusNotStarted;
            }
 
        if (data.PackedB) {
            MlasGetSumValBufFromPackedMutable<uint8_t>(data.PackedB, N, SumBuffer, Packed);
            for (size_t n = 0; n < N; n += StrideN) {
                uint64_t* PackStatus = (uint64_t*)(SumBuffer + n);
                *PackStatus = PackStatusNotStarted;
            }
        }
    }

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        auto Data = &DataParams[gemm_i];
        auto WorkBlock = &WorkBlocks[gemm_i];

        const ptrdiff_t ThreadIdM = blk_i / ThreadCountN;
        const ptrdiff_t ThreadIdN = blk_i % ThreadCountN;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = StrideM;

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = StrideN;

        GemmU8X8Operation(&Shape, Data, WorkBlock, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}


size_t
MLASCALL
MlasGemmPackBSize(
    size_t N,
    size_t K,
    bool BIsSigned
    )
/*++

Routine Description:

    This routine computes the number of bytes required to pack a matrix with
    the supplied shape and type.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the the number of rows of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

Return Value:

    Returns the number of bytes required to pack the matrix, else zero if the
        current implementation does not support packing.

--*/
{
    //
    // Retrieve the packing parameters.
    //

    const auto* GemmU8X8Dispatch = MlasGemmU8X8GetDispatch(BIsSigned);

    size_t PackedK = GemmU8X8Dispatch->PackedK;
    size_t PackedStrideK = GemmU8X8Dispatch->PackedStrideK;

    if (PackedStrideK == 0) {
        return 0;
    }

    //
    // Compute the number of bytes required to hold the packed buffer.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);

    const size_t BytesRequired =
        (AlignedN * sizeof(int32_t)) + (AlignedN * AlignedK * sizeof(uint8_t));
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired = (BytesRequired + BufferAlignment - 1) &
        ~(BufferAlignment - 1);

    return AlignedBytesRequired;
}

void
MLASCALL
MlasGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,
    bool BIsSigned,
    void* PackedB
    )
/*++

Routine Description:

    This routine packs the supplied matrix B to the supplied packed matrix B
    buffer. The size of the packed buffer was obtained from MlasGemmPackBSize.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the the number of rows of matrix B.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

    PackedB - Supplies the address of packed matrix B.

Return Value:

    None.

--*/
{
    //
    // Retrieve the packing parameters.
    //

    const auto* GemmU8X8Dispatch = MlasGemmU8X8GetDispatch(BIsSigned);

    size_t PackedK = GemmU8X8Dispatch->PackedK;

    //
    // Reserve and initialize storage for the column sum buffer to hold the sums
    // of the elements along each of the columns.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);

    int32_t* PackedColumnSumBuffer = (int32_t*)PackedB;
    std::fill_n(PackedColumnSumBuffer, AlignedN, 0);
    PackedB = PackedColumnSumBuffer + AlignedN;

        //
        // Step through each slice of matrix B along the N dimension.
        //

        const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);
        uint8_t* pb = (uint8_t*)PackedB;
        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            constexpr size_t BatchedN = 128;
            MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[BatchedN], 64);

            CountN = std::min(N - n, BatchedN);

            GemmU8X8Dispatch->CopyPackBRoutine(pb, B + n, ldb, CountN, K, ColumnSumBuffer, BIsSigned);

            //
            // Accumulate this batch of the column sum buffer into the packed
            // buffer accumulators.
            //

            for (size_t nn = 0; nn < CountN; nn++) {
                PackedColumnSumBuffer[n + nn] += ColumnSumBuffer[nn];
            }

            pb += CountN * AlignedK;
        }

}
