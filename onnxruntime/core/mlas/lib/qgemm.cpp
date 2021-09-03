/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm.cpp

Abstract:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// Define the parameters to execute segments of a QGEMM operation on worker
// threads.
//

struct MLAS_GEMM_U8X8_WORK_BLOCK {
    ptrdiff_t ThreadCountM;
    ptrdiff_t ThreadCountN;
    ptrdiff_t StrideM;
    ptrdiff_t StrideN;
};

void
MlasGemmU8X8Threaded(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    ptrdiff_t ThreadId
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    QGEMM operation.

Arguments:

    ThreadInfo - Supplies the structure containing the thread task partition info.

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const ptrdiff_t ThreadIdM = ThreadId / WorkBlock->ThreadCountN;
    const ptrdiff_t ThreadIdN = ThreadId % WorkBlock->ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //

    size_t RangeStartM;
    size_t RangeCountM;

    const size_t M = Shape->M;

    MlasPartitionWork(ThreadIdM, WorkBlock->ThreadCountM, M, &RangeStartM, &RangeCountM);

    //
    // Partition the operation along the N dimension.
    //

    size_t RangeStartN;
    size_t RangeCountN;

    const size_t N = Shape->N;

    const size_t BlockedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) /
        MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, WorkBlock->ThreadCountN, BlockedN,
        &RangeStartN, &RangeCountN);

    RangeStartN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    RangeCountN = std::min(N - RangeStartN, RangeCountN);

    //
    // Dispatch the partitioned operation.
    //

    const auto* GemmU8X8Dispatch = MlasGemmU8X8GetDispatch(Shape->BIsSigned);
    MLAS_GEMM_U8X8_OPERATION* GemmU8X8Operation;

    if (Data->BIsPacked) {
        GemmU8X8Operation = GemmU8X8Dispatch->PackedOperation;
    } else {
        GemmU8X8Operation = GemmU8X8Dispatch->Operation;
    }

    GemmU8X8Operation(Shape, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
}


void
MLASCALL
MlasGemmBatch(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool)
{
    //
    // Locate the QGEMM dispatch block.
    //

    const auto* GemmU8X8Dispatch = MlasGemmU8X8GetDispatch(Shape.BIsSigned);

    //
    // Compute the number of target threads given the complexity of the QGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const ptrdiff_t M = Shape.M;
    const ptrdiff_t N = Shape.N;
    const ptrdiff_t K = Shape.K;
    const double Complexity = double(M) * double(N) * double(K);

    ptrdiff_t ThreadsPerGemm;

    if (Complexity < double(MLAS_QGEMM_THREAD_COMPLEXITY * MlasPlatform.MaximumThreadCount)) {
        ThreadsPerGemm = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        ThreadsPerGemm = MlasPlatform.MaximumThreadCount;
    }

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (ThreadsPerGemm >= MaximumThreadCount) {
        ThreadsPerGemm = MaximumThreadCount;
    }

    MLAS_GEMM_U8X8_WORK_BLOCK WorkBlock{1, 1, M, N};

    if (ThreadsPerGemm <= 1) {
        ThreadsPerGemm = 1;
    } else {
        //
        // Segment the operation across multiple threads.
        //

        if ((N * ThreadsPerGemm) <= (M + M / 8)) {
            //
            // tall and thin, one dimensional partition
            //

            const ptrdiff_t BlockedM = (M + MLAS_QGEMM_M_ALIGN - 1) / MLAS_QGEMM_M_ALIGN;
            WorkBlock.StrideM =
                ((BlockedM + ThreadsPerGemm - 1) / ThreadsPerGemm) * MLAS_QGEMM_M_ALIGN;
            WorkBlock.ThreadCountM = (M + WorkBlock.StrideM - 1)/ WorkBlock.StrideM;
        }
        else if ((M * ThreadsPerGemm) <= (N + N / 8)) {
            //
            // short and flat, one  dimensional partition
            //

            const ptrdiff_t BlockedN =
                (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) / MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
            WorkBlock.StrideN = ((BlockedN + ThreadsPerGemm - 1) / ThreadsPerGemm) *
                                MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
            WorkBlock.ThreadCountN = (N + WorkBlock.StrideN - 1) / WorkBlock.StrideN;
        }
        else {
            //
            // Try to partition the resulting matrix into shapes such that N is about 2*M,
            //

            const double sq_area = double(M) * double(N) / ThreadsPerGemm;
            const double sq_edge = sqrt(sq_area * 2);

            ptrdiff_t strideN = ptrdiff_t(ceil(sq_edge / MLAS_QGEMM_STRIDEN_THREAD_ALIGN)) *
                                MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
            WorkBlock.ThreadCountN = (N + strideN - 1) / strideN;
            WorkBlock.StrideN = strideN;

            ptrdiff_t strideM =
                (ptrdiff_t)ceil(sq_area / (strideN * MLAS_QGEMM_M_ALIGN)) * MLAS_QGEMM_M_ALIGN;
            WorkBlock.ThreadCountM = (M + strideM - 1) / strideM;
            WorkBlock.StrideM = strideM;
        }
        ThreadsPerGemm = WorkBlock.ThreadCountM * WorkBlock.ThreadCountN;
    }

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN,
                          [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;

        auto Data = &DataParams[gemm_i];
        MLAS_GEMM_U8X8_OPERATION* GemmU8X8Operation;
        if (Data->BIsPacked) {
            GemmU8X8Operation = GemmU8X8Dispatch->PackedOperation;
        } else {
            GemmU8X8Operation = GemmU8X8Dispatch->Operation;
        }

        const ptrdiff_t ThreadIdM = blk_i / WorkBlock.ThreadCountN;
        const ptrdiff_t ThreadIdN = blk_i % WorkBlock.ThreadCountN;

        const size_t RangeStartM = ThreadIdM * WorkBlock.StrideM;
        const size_t RangeCountM = std::min(Shape.M - RangeStartM, (size_t)WorkBlock.StrideM);

        const size_t RangeStartN = ThreadIdN * WorkBlock.StrideN;
        const size_t RangeCountN = std::min(Shape.N - RangeStartN, (size_t)WorkBlock.StrideN);

        GemmU8X8Operation(&Shape, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
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
    size_t PackedStrideK = GemmU8X8Dispatch->PackedStrideK;

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
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, PackedStrideK);

        //
        // Step through each slice of matrix B along the N dimension.
        //

        const size_t AlignedK = (CountK + PackedK - 1) & ~(PackedK - 1);
        uint8_t* pb = (uint8_t*)PackedB;
        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            constexpr size_t BatchedN = 128;
            MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[BatchedN], 64);

            CountN = std::min(N - n, BatchedN);

            GemmU8X8Dispatch->CopyPackBRoutine(pb, B + n, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);

            //
            // Accumulate this batch of the column sum buffer into the packed
            // buffer accumulators.
            //

            for (size_t nn = 0; nn < CountN; nn++) {
                PackedColumnSumBuffer[n + nn] += ColumnSumBuffer[nn];
            }

            pb += CountN * AlignedK;
        }

        PackedB = (uint8_t*)PackedB + AlignedN * AlignedK;
        B += ldb * CountK;
    }
}
