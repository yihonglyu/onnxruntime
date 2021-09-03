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

#include <stdexcept>
#include "mlasi.h"

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

class Formatter
{
   public:
    Formatter() {}
    ~Formatter() {}

    template <typename Type>
    Formatter& operator<<(const Type& value)
    {
        stream_ << value;
        return *this;
    }

    std::string str() const { return stream_.str(); }
    operator std::string() const { return stream_.str(); }

    enum ConvertToString { to_str };
    std::string operator>>(ConvertToString) { return stream_.str(); }

   private:
    std::stringstream stream_;

    Formatter(const Formatter&);
    Formatter& operator=(Formatter&);
};

// TODO!! Block M should be aligned according to QGEMM kernel dimension
constexpr ptrdiff_t MLAS_QGEMM_M_ALIGN = 4;

template<typename KernelType>
void
MlasComputePackBufLayout(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    MLAS_GEMM_U8X8_STRIDES& Strides,
    typename KernelType::PackedAType*& PackBufA,
    typename KernelType::PackedBType** PackBufB,
    int32_t*& RowSumBuf,
    int32_t*& ColumnSumBuf,
    int32_t*& ZeroPointBBuf
    )
{
    //
    // The plan is to allocate about half of the cache to B buffer,
    // and about a quarter to A buffer. This is because in our operation
    // methods, we iterate N in the outer loop and M in the inner loop.
    // While we pack A in the inner loop, it reads from original matrix
    // and writes to the packing buffer, taking twice the space.
    // If A buffer is too big, it would evict B buffer out of the cache.
    //

    constexpr MLAS_GEMM_U8X8_STRIDES ConstStrides = KernelType::PackedStrides;
    const size_t CacheSize = MlasGetL2CacheSizePerCore();

    const size_t AlignedK = (CountK + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) &
                ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    Strides.K = std::min(ConstStrides.K, AlignedK);

    // leave room for row col sum buffer when computing stride M or stride N
    constexpr size_t RowColSumSpacerFactor = 8;
    size_t MplusN = CacheSize / (Strides.K + RowColSumSpacerFactor);

    if (sizeof(typename KernelType::PackedAType) > 1 ||
        sizeof(typename KernelType::PackedBType) > 1) {
        // over-simplistic approach to deal with int16 packing type, needs improvment
        MplusN /= 2;
    }

    size_t StrideN = std::min(MplusN / 2, CountN);
    size_t StrideM = (MplusN - StrideN) / 2;

    if (StrideM < CountM) {
        // try to  distribute stride M evenly
        const size_t msteps = (CountM + StrideM - 1) / StrideM;
        StrideM = (CountM + msteps - 1) / msteps;

        StrideM = (StrideM + MLAS_QGEMM_M_ALIGN - 1) & ~(MLAS_QGEMM_M_ALIGN - 1);
        if (StrideM > MLAS_QGEMM_M_ALIGN && StrideM > ((MplusN - StrideN) / 2)) {
            StrideM -= MLAS_QGEMM_M_ALIGN;
        }
    } else {
        StrideM = CountM;
    }

    // re-adjust N based on M
    StrideN = (MplusN - StrideM * 2);
    if (StrideN < CountN) {
        const size_t nsteps = (CountN + StrideN - 1) / StrideN;
        StrideN = (CountN + nsteps - 1) / nsteps;
    } else {
        StrideN = CountN;
    }
    StrideN =
        (StrideN + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    if (StrideN > MLAS_QGEMM_STRIDEN_THREAD_ALIGN && StrideN > (MplusN - StrideM * 2)) {
        StrideN -= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
    }
    
    Strides.M = StrideM;
    Strides.N = StrideN;
    
    const size_t AlignedM =
        (StrideM + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);

    ptrdiff_t addr = (ptrdiff_t)MlasGetThreadPackBuf();
    size_t bytes = 0;

    RowSumBuf = (int32_t*)(addr + bytes);
    bytes += AlignedM * sizeof(int32_t);

    PackBufA = reinterpret_cast<typename KernelType::PackedAType*>(addr + bytes);
    size_t Abytes = StrideM * Strides.K * sizeof(typename KernelType::PackedAType);
    Abytes = (Abytes + 63) & ~63;
    bytes += Abytes;

    ColumnSumBuf = (int32_t*)(addr + bytes);
    bytes += StrideN * sizeof(int32_t);

    ZeroPointBBuf = (int32_t*)(addr + bytes);
    bytes += StrideN * sizeof(int32_t);
    
    if (nullptr != PackBufB) {
        *PackBufB = reinterpret_cast<typename KernelType::PackedBType*>(addr + bytes);
    }
    bytes += StrideN * Strides.K * sizeof(typename KernelType::PackedBType);
    if (bytes > (CacheSize + 64 * 5)) {
        throw std::runtime_error(
            Formatter() << "Internal Error: MLAS packing buffer overflow! M: " << CountM
                        << ", N: " << CountN << ", K: " << CountK << ", Cache Size: " << CacheSize
                        << ", StrideM: " << StrideM << ", StrideN: " << StrideN
                        << ", StrideK: " << Strides.K << ", Used bytes: " << bytes);
    }
}

template<typename KernelType>
void
MlasGemmU8X8Operation(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    RangeStartM - Supplies the starting row index to output.

    RangeCountM - Supplies the number of rows to output.

    RangeStartN - Supplies the starting column index to output.

    RangeCountN - Supplies the number of columns to output.

Return Value:

    None.

--*/
{
    MLAS_GEMM_U8X8_STRIDES Strides;
    typename KernelType::PackedAType* PanelA;
    typename KernelType::PackedBType* PanelB;

    int32_t* RowSumBuffer;
    int32_t* ColumnSumBuffer;
    int32_t* ZeroPointBBuffer;

    const size_t K = Shape->K;

    MlasComputePackBufLayout<KernelType>(RangeCountM, RangeCountN, K, Strides, PanelA, &PanelB,
                                         RowSumBuffer, ColumnSumBuffer, ZeroPointBBuffer);

    const size_t lda = Data->lda;
    const size_t ldb = Data->ldb;
    const size_t ldc = Data->ldc;

    const uint8_t* A = Data->A + RangeStartM * lda;
    const uint8_t* B = (const uint8_t*)Data->B + RangeStartN;
    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB = Data->PerColumnZeroPoints ?
        Data->ZeroPointB + RangeStartN : nullptr;
    bool IsAccumulateMode = Shape->IsAccumulateMode;

    int32_t ZeroPointA = Data->ZeroPointA;
    int32_t ZeroPointB = typename KernelType::OffsetBType(*Data->ZeroPointB);

    //
    // Try to use a GEMV kernel if supported by this kernel type.
    //

    if ((RangeCountM == 1) &&
        (ZeroPointA == 0) && (PackedZeroPointB == nullptr) && (ZeroPointB == 0) &&
        (Data->OutputProcessor == nullptr)) {
        if (MlasGemmU8X8TryGemvKernel<KernelType>(A, B, ldb, C, K, RangeCountN, Shape->BIsSigned)) {
            return;
        }
    }

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
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {

            CountN = std::min(RangeCountN - n, Strides.N);

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmU8X8FixupZeroPointB<KernelType>(
                    PackedZeroPointB + n,
                    ZeroPointBBuffer,
                    CountN,
                    Shape->BIsSigned);
            }

            //
            // Copy a panel of matrix B to a local packed buffer.
            //

            MlasGemmU8X8CopyPackB<KernelType>(
                PanelB,
                B + n,
                ldb,
                CountN,
                CountK,
                ColumnSumBuffer,
                Shape->BIsSigned);

            MlasGemmU8X8ScaleSumBuffer(ColumnSumBuffer, CountN, -ZeroPointA);

            //
            // Step through each slice of matrix A along the M dimension.
            //

            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < RangeCountM; m += CountM) {

                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                MlasGemmU8X8CopyPackA<KernelType>(
                    PanelA,
                    A + m * lda,
                    lda,
                    CountM,
                    CountK,
                    RowSumBuffer);

                //
                // Apply the global depth value constant without the ZeroPointB scaling from:
                //
                //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                //              ==>
                //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
                //
                // The ZeroPointB term is factored out and either applied below for per-matrix
                // quantization or inside the kernel for per-column quantization.
                //

                for (size_t mm = 0; mm < CountM; mm++) {
                    RowSumBuffer[mm] -= int32_t(CountK) * ZeroPointA;
                }

                //
                // Scale the row sums by the per-matrix zero point offset of matrix B.
                //

                if (PackedZeroPointB == nullptr) {
                    MlasGemmU8X8ScaleSumBuffer(RowSumBuffer, CountM, -ZeroPointB);
                }

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0) && !IsAccumulateMode;
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled = MlasGemmU8X8Kernel<KernelType>(
                        pa,
                        PanelB,
                        c,
                        PackedCountK,
                        RowsRemaining,
                        CountN,
                        ldc,
                        RowSums,
                        ColumnSumBuffer,
                        (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr,
                        ZeroMode);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C,
                            RangeStartM + m + CountM - RowsRemaining,
                            RangeStartN + n,
                            RowsHandled,
                            CountN,
                            Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        B += CountK * ldb;
    }
}


template<typename KernelType>
void
MlasGemmU8X8PackedOperation(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    RangeStartM - Supplies the starting row index to output.

    RangeCountM - Supplies the number of rows to output.

    RangeStartN - Supplies the starting column index to output.

    RangeCountN - Supplies the number of columns to output.

Return Value:

    None.

--*/
{
    MLAS_GEMM_U8X8_STRIDES Strides;
    typename KernelType::PackedAType* PanelA;

    int32_t* RowSumBuffer;
    int32_t* ColumnSumBuffer;
    int32_t* ZeroPointBBuffer;

    const size_t K = Shape->K;
    MlasComputePackBufLayout<KernelType>(RangeCountM, RangeCountN, K, Strides, PanelA, nullptr,
                                         RowSumBuffer, ColumnSumBuffer, ZeroPointBBuffer);

    const size_t lda = Data->lda;
    const size_t ldc = Data->ldc;

    const uint8_t* A = Data->A + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)Data->B;
    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB = Data->PerColumnZeroPoints ?
        Data->ZeroPointB + RangeStartN : nullptr;
    bool IsAccumulateMode = Shape->IsAccumulateMode;

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
    // Extract the pointer to the column sum buffer from the packed matrix.
    //

    const size_t AlignedN =
        (Shape->N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const int32_t* PackedColumnSumBuffer = (const int32_t*)PackedB;
    PackedB = (const uint8_t*)(PackedColumnSumBuffer + AlignedN);
    PackedColumnSumBuffer += RangeStartN;

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        if (k > 0) {
            std::fill_n(ColumnSumBuffer, Strides.N, 0);
        }

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {

            CountN = std::min(RangeCountN - n, Strides.N);

            if (k == 0) {
                MlasGemmU8X8ScaleSumBuffer(ColumnSumBuffer, PackedColumnSumBuffer + n,
                    CountN, -ZeroPointA);
            }

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmU8X8FixupZeroPointB<KernelType>(
                    PackedZeroPointB + n,
                    ZeroPointBBuffer,
                    CountN,
                    Shape->BIsSigned);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const uint8_t* b = PackedB + (RangeStartN + n) *
                KernelType::PackedK * PackedCountK;
            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < RangeCountM; m += CountM) {

                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                MlasGemmU8X8CopyPackA<KernelType>(
                    PanelA,
                    A + m * lda,
                    lda,
                    CountM,
                    CountK,
                    RowSumBuffer);

                //
                // Apply the global depth value constant without the ZeroPointB scaling from:
                //
                //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                //              ==>
                //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
                //
                // The ZeroPointB term is factored out and either applied below for per-matrix
                // quantization or inside the kernel for per-column quantization.
                //

                for (size_t mm = 0; mm < CountM; mm++) {
                    RowSumBuffer[mm] -= int32_t(CountK) * ZeroPointA;
                }

                //
                // Scale the row sums by the per-matrix zero point offset of matrix B.
                //

                if (PackedZeroPointB == nullptr) {
                    MlasGemmU8X8ScaleSumBuffer(RowSumBuffer, CountM, -ZeroPointB);
                }

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0) && !IsAccumulateMode;
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled = MlasGemmU8X8Kernel<KernelType>(
                        pa,
                        b,
                        c,
                        PackedCountK,
                        RowsRemaining,
                        CountN,
                        ldc,
                        RowSums,
                        ColumnSumBuffer,
                        (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr,
                        ZeroMode);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C,
                            RangeStartM + m + CountM - RowsRemaining,
                            RangeStartN + n,
                            RowsHandled,
                            CountN,
                            Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        PackedB = (const uint8_t*)PackedB + AlignedN * CountK;
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
    MLAS_GEMM_U8X8_OPERATION* PackedOperation;
    MLAS_GEMM_U8X8_COPY_PACKB_ROUTINE* CopyPackBRoutine;
    size_t PackedK;
    size_t PackedStrideK;
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
