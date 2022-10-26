/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8S8KernelAmx.s

Abstract:

    This module implements the packing functions for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    These packing functions are suited for AMX Qgemm kernel. The implementation only
    uses AVX2 instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

//
// Stack frame layout for the U8S8 CopyPackB routine.
//

        .equ    .LGemmU8S8CopyPackBFrame_PaddedMatrixBData, -72
        .equ    .LGemmU8S8CopyPackBFrame_Padding, -8
        .equ    .LGemmU8S8CopyPackBFrame_SavedR12, 0
        .equ    .LGemmU8S8CopyPackBFrame_SavedRbx, 8
        .equ    .LGemmU8S8CopyPackBFrame_SavedRbp, 16
        .equ    .LGemmU8S8CopyPackBFrame_ReturnAddress, 24
        .equ    .LGemmU8S8CopyPackBFrame_BIsSigned, 32

        .text

/*++

Routine Description:

    This routine copies elements from the source B matrix to the destination
    packed buffer.

    This implementation is almost identical to MlasGemmU8S8CopyPackBAvx2
    where it traverse B vertically, take a block of 4 row 16 col, transpose
    and store it, then go down 4 row to grab the next 4x16 block. The only
    difference here is that we need K to be aligned to 64 to the fill
    an AMX tile.

Arguments:

    D (rdi) - Supplies the address of the destination packed buffer.

    B (rsi) - Supplies the address of the source matrix.

    ldb (rdx) - Supplies the number of elements per row of the source matrix.

    CountN (rcx) - Supplies the number of columns of the source matrix to copy.

    CountK (r8) - Supplies the number of rows of the source matrix to copy.

    ColumnSumBuffer (r9) - Supplies the address of the buffer to receive the sums
        of the elements along each of the columns.

    BIsSigned - Supplies true if the source matrix is signed data, else false if
        the source matrix is unsigned data.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasGemmU8S8CopyPackBAmx

        push    rbp
        push    rbx
        push    r12

        mov     r10,rdx
        lea     r11,[r10+r10*2]             # compute ldb * 3
        lea     r12,[r8+3]                  # compute extra padding for 64|K
        shr     r12,2
        neg     r12
        and     r12,15
        vpcmpeqw ymm7,ymm7,ymm7             # generate word vector [0xFFFF]
        vpsrlw  ymm7,ymm7,15                # generate word vector [0x0001]
        vpsllw  ymm8,ymm7,8                 # generate word vector [0x0100]
        vpor    ymm8,ymm7,ymm8              # generate word vector [0x0101]

//
// Compute the bit flip vector to adjust input from U8 to S8.
//

        vpxor   xmm9,xmm9,xmm9              # generate word vector [0x0000]
        cmp     BYTE PTR .LGemmU8S8CopyPackBFrame_BIsSigned[rsp],0
        jnz     .LCopyPackB.SkipUnsignedBitFlipVector
        vpsllw  ymm9,ymm8,7                 # generate word vector [0x8080]

.LCopyPackB.SkipUnsignedBitFlipVector:

//
// Process 16 columns of matrix B in a loop.
//

        sub     rcx,16
        jb      .LCopyPackB.ProcessRemainingColumns

.LCopyPackB.ProcessNextColumnN16:
        vpxor   xmm0,xmm0,xmm0              # clear column accumulators
        vpxor   xmm1,xmm1,xmm1
        mov     rdx,rsi
        add     rsi,16                      # advance next matrix B by 16 columns
        mov     rbx,r8                      # reload rows remaining
        sub     rbx,4
        jb      .LCopyPackB.ProcessRemainingRowsN16

.LCopyPackB.ProcessNextRowLoopN16:
        vmovdqu xmm2,XMMWORD PTR [rdx]      # load 4 rows
        vmovdqu xmm3,XMMWORD PTR [rdx+r10]
        vmovdqu xmm4,XMMWORD PTR [rdx+r10*2]
        vmovdqu xmm5,XMMWORD PTR [rdx+r11]
        lea     rdx,[rdx+r10*4]             # advance matrix B by 4 rows

.LCopyPackB.InterleaveRowDataN16:
        vpunpcklbw xmm6,xmm2,xmm3           # interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vpunpcklbw xmm2,xmm4,xmm5
        vpunpckhbw xmm5,xmm4,xmm5
        vpunpcklwd xmm4,xmm6,xmm2
        vpunpckhwd xmm6,xmm6,xmm2
        vpunpcklwd xmm2,xmm3,xmm5
        vpunpckhwd xmm3,xmm3,xmm5
        vinserti128 ymm4,ymm4,xmm6,1
        vinserti128 ymm2,ymm2,xmm3,1
        vpxor   ymm4,ymm4,ymm9              # optionally adjust unsigned data
        vpxor   ymm2,ymm2,ymm9
        vmovdqu YMMWORD PTR [rdi],ymm4      # store interleaved rows
        vmovdqu YMMWORD PTR [rdi+32],ymm2
        vpmaddubsw ymm4,ymm8,ymm4           # horizontal byte+byte=word per row
        vpmaddwd ymm4,ymm4,ymm7             # horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              # accumulate per column
        vpmaddubsw ymm2,ymm8,ymm2
        vpmaddwd ymm2,ymm2,ymm7
        vpaddd  ymm1,ymm1,ymm2
        add     rdi,64                      # advance matrix D by 64 bytes
        sub     rbx,4                       # subtract rows remaining
        jae     .LCopyPackB.ProcessNextRowLoopN16

//
// Process the less than 4 remaining rows where the row has 16 columns.
//

.LCopyPackB.ProcessRemainingRowsN16:
        add     rbx,4                       # correct for over-subtract above
        jz      .LCopyPackB.StoreColumnSumBufferN16
        vmovdqu xmm2,XMMWORD PTR [rdx]
        vmovaps xmm3,xmm9
        vmovaps xmm4,xmm9
        vmovaps xmm5,xmm9
        xor     ebx,ebx                     # no more rows remaining
        test    r8b,2                       # (CountK & 2) != 0?
        jz      .LCopyPackB.InterleaveRowDataN16
        vmovdqu xmm3,XMMWORD PTR [rdx+r10]
        test    r8b,1                       # (CountK & 1) != 0?
        jz      .LCopyPackB.InterleaveRowDataN16
        vmovdqu xmm4,XMMWORD PTR [rdx+r10*2]
        jmp     .LCopyPackB.InterleaveRowDataN16

.LCopyPackB.StoreColumnSumBufferN16:
        vmovdqu YMMWORD PTR [r9],ymm0
        vmovdqu YMMWORD PTR [r9+32],ymm1
        test    r12,r12
        jz      .LCopyPackB.N16K64PaddingFinished
        mov     rax, r12
        vpxor   xmm0,xmm0,xmm0
.LCopyPackB.N16K64Padding:
        vmovdqu YMMWORD PTR [rdi],ymm0      # store 0
        vmovdqu YMMWORD PTR [rdi+32],ymm0
        add     rdi,64
        dec     rax
        jnz     .LCopyPackB.N16K64Padding
.LCopyPackB.N16K64PaddingFinished:
        add     r9,16*4                     # advance column sum buffer by 16 dwords
        sub     rcx,16                      # subtract columns remaining
        jae     .LCopyPackB.ProcessNextColumnN16

.LCopyPackB.ProcessRemainingColumns:
        add     rcx,16                      # correct for over-subtract above
        jnz     .LCopyPackB.ProcessColumnNUnaligned

//
// Restore non-volatile registers and return.
//

.LCopyPackB.ExitRoutine:
        vzeroupper

        pop     r12
        pop     rbx
        pop     rbp
        ret

//
// Process the remaining columns of matrix B.
//

.LCopyPackB.ProcessColumnNUnaligned:
        vpxor   xmm0,xmm0,xmm0              # clear column accumulators
        vpxor   xmm1,xmm1,xmm1
        vmovdqu YMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp],ymm9
        vmovdqu YMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp+32],ymm9
        sub     r8,4
        jb      .LCopyPackB.ProcessRemainingRowsNUnaligned

.LCopyPackB.ProcessNextRowLoopNUnaligned:
        mov     rdx,rsi
        lea     rbp,.LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp]
        test    cl,8                        # (CountN & 8) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan8K4
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r10]
        mov     QWORD PTR [rbp+16],rax
        mov     rax,QWORD PTR [rdx+r10*2]
        mov     QWORD PTR [rbp+32],rax
        mov     rax,QWORD PTR [rdx+r11]
        mov     QWORD PTR [rbp+48],rax
        add     rdx,8                       # advance matrix B
        add     rbp,8                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan8K4:
        test    cl,4                        # (CountN & 4) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan4K4
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r10]
        mov     DWORD PTR [rbp+16],eax
        mov     eax,DWORD PTR [rdx+r10*2]
        mov     DWORD PTR [rbp+32],eax
        mov     eax,DWORD PTR [rdx+r11]
        mov     DWORD PTR [rbp+48],eax
        add     rdx,4                       # advance matrix B
        add     rbp,4                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan4K4:
        test    cl,2                        # (CountN & 2) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan2K4
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r10]
        mov     WORD PTR [rbp+16],ax
        movzx   eax,WORD PTR [rdx+r10*2]
        mov     WORD PTR [rbp+32],ax
        movzx   eax,WORD PTR [rdx+r11]
        mov     WORD PTR [rbp+48],ax
        add     rdx,2                       # advance matrix B
        add     rbp,2                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan2K4:
        test    cl,1                        # (CountN & 1) != 0?
        jz      .LCopyPackB.ProcessPaddedMatrixBData
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r10]
        mov     BYTE PTR [rbp+16],al
        movzx   eax,BYTE PTR [rdx+r10*2]
        mov     BYTE PTR [rbp+32],al
        movzx   eax,BYTE PTR [rdx+r11]
        mov     BYTE PTR [rbp+48],al

.LCopyPackB.ProcessPaddedMatrixBData:
        vmovdqu xmm2,XMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp]
        vmovdqu xmm3,XMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp+16]
        vmovdqu xmm4,XMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp+32]
        vmovdqu xmm5,XMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp+48]
        vpunpcklbw xmm6,xmm2,xmm3           # interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vpunpcklbw xmm2,xmm4,xmm5
        vpunpckhbw xmm5,xmm4,xmm5
        vpunpcklwd xmm4,xmm6,xmm2
        vpunpckhwd xmm6,xmm6,xmm2
        vpunpcklwd xmm2,xmm3,xmm5
        vpunpckhwd xmm3,xmm3,xmm5
        vinserti128 ymm4,ymm4,xmm6,1
        vinserti128 ymm2,ymm2,xmm3,1
        vpxor   ymm4,ymm4,ymm9              # optionally adjust unsigned data
        vpxor   ymm2,ymm2,ymm9
        vmovdqu YMMWORD PTR [rdi],ymm4      # store interleaved rows
        vmovdqu YMMWORD PTR [rdi+32],ymm2
        vpmaddubsw ymm4,ymm8,ymm4           # horizontal byte+byte=word per row
        vpmaddwd ymm4,ymm4,ymm7             # horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              # accumulate per column
        vpmaddubsw ymm2,ymm8,ymm2
        vpmaddwd ymm2,ymm2,ymm7
        vpaddd  ymm1,ymm1,ymm2
        lea     rsi,[rsi+r10*4]             # advance next matrix B by 4 rows
        add     rdi,64                      # advance matrix D by 64 bytes
        sub     r8,4                        # subtract rows remaining
        jae     .LCopyPackB.ProcessNextRowLoopNUnaligned

.LCopyPackB.ProcessRemainingRowsNUnaligned:
        add     r8,4
        jz      .LCopyPackB.StoreColumnSumBufferNUnaligned

//
// Process the less than 4 remaining rows where the row has less than 16 columns.
//

        lea     rbp,.LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp]
        vmovdqu YMMWORD PTR [rbp],ymm9
        vmovdqu YMMWORD PTR [rbp+32],ymm9

.LCopyPackB.CopyUnalignedRowLoop:
        lea     r11,[rbp+16]                # advance next padded buffer by 16 bytes
        mov     rdx,rsi
        test    cl,8                        # (CountN & 8) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan8KSmall
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        add     rdx,8                       # advance matrix B
        add     rbp,8                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan8KSmall:
        test    cl,4                        # (CountN & 4) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan4KSmall
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        add     rdx,4                       # advance matrix B
        add     rbp,4                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan4KSmall:
        test    cl,2                      # (CountN & 2) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan2KSmall
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        add     rdx,2                       # advance matrix B
        add     rbp,2                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan2KSmall:
        test    cl,1                        # (CountN & 1) != 0?
        jz      .LCopyPackB.DoneCopyRemainingCountNKSmall
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al

.LCopyPackB.DoneCopyRemainingCountNKSmall:
        dec     r8
        jz      .LCopyPackB.ProcessPaddedMatrixBData
        add     rsi,r10                     # advance next matrix B by 1 row
        mov     rbp,r11
        jmp     .LCopyPackB.CopyUnalignedRowLoop

.LCopyPackB.StoreColumnSumBufferNUnaligned:
        vmovdqu YMMWORD PTR [r9],ymm0
        vmovdqu YMMWORD PTR [r9+32],ymm1
        test    r12,r12
        jz      .LCopyPackB.ExitRoutine
        mov     rax, r12
        vpxor   xmm0,xmm0,xmm0
.LCopyPackB.K64Padding:
        vmovdqu YMMWORD PTR [rdi],ymm0      # store 0
        vmovdqu YMMWORD PTR [rdi+32],ymm0
        add     rdi,64
        dec     rax
        jne     .LCopyPackB.K64Padding
        jmp     .LCopyPackB.ExitRoutine

        .end
