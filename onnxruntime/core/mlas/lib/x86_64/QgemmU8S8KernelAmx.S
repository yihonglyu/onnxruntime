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


//
// Stack frame layout for the U8S8 CopyPackA routine.
//
        .equ    .LGemmU8S8CopyPackAFrame_PaddedMatrixAData, -72
        .equ    .LGemmU8S8CopyPackAFrame_Padding, -8
        .equ    .LGemmU8S8CopyPackAFrame_SavedR13, 0
        .equ    .LGemmU8S8CopyPackAFrame_SavedR12, 8
        .equ    .LGemmU8S8CopyPackAFrame_SavedRbx, 16
        .equ    .LGemmU8S8CopyPackAFrame_SavedRbp, 24
        .equ    .LGemmU8S8CopyPackAFrame_ReturnAddress, 32

/*++

Routine Description:

    This routine copies elements from the source matrix A to the destination
    packed buffer.

Arguments:

    D (rdi) - Supplies the address of the destination packed buffer.

    A (rsi) - Supplies the address of the source matrix.

    lda (rdx) - Supplies the number of elements per row of the source matrix.

    CountM (rcx) - Supplies the number of rows of the source matrix to copy.

    CountK (r8) - Supplies the number of columns of the source matrix to copy.

    RowSumBuffer (r9) - Supplies the address of the buffer to receive the sums
        of the elements along each of the rows.
        by the zero point offset.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasGemmU8S8CopyPackAAmx

        push    rbp
        push    rbx
        push    r12
        push    r13

        mov         r10,rdx                     # lda
        mov         r11,rcx                     # m = CountM
        lea         r12,[r8+63]
        and         r12,NOT 63                  # align CountK up to 64
        vpternlogd  zmm8,zmm8,zmm8,255          # generate word vector [0xFFFF]
        vpsrlw      zmm8,zmm8,15                # generate word vector [0x0001]
        vpsllw      zmm9,zmm8,8                 # generate word vector [0x0100]
        vpord       zmm9,zmm8,zmm9              # generate word vector [0x0101]
        lea         r13,[r10+r10*2]             # compute ldb * 3
        lea         rax,[r12+r12*2]             # compute AlignedCountK * 3

//
// Compute the conditional load mask for an unaligned CountK.
//

        mov     ecx,r8d                         # CountK
        neg     ecx
        and     ecx,63
        mov     rbx,-1
        shr     rbx,cl                          # mask for left over k < 64
        kmovq   k1,rbx                          # mask

//
// Process 4 rows of matrix A in a loop.
//

        sub     r11,4                           # m -= 4
        jb      .LCopyPackA.ProcessRemainingRows

.LCopyPackA.ProcessNextRowM4:
        vpxor   xmm0,xmm0,xmm0                  # clear row accumulators
        vpxor   xmm1,xmm1,xmm1
        vpxor   xmm2,xmm2,xmm2
        vpxor   xmm3,xmm3,xmm3
        mov     rdx,rsi                         # src = A
        mov     rcx,rdi                         # dst = D
        lea     rsi,[rsi+r10*4]                 # advance next matrix A by 4 rows
        lea     rdi,[rdi+r12*4]                 # advance next matrix D by 4 rows
        mov     rbx,r8                          # k = CountK
        sub     rbx,64
        jb      .LCopyPackA.ProcessRemainingColumnsM4

.LCopyPackA.ProcessNextColumnLoopM4:
        vmovdqu64  zmm4,ZMMWORD PTR [rdx]
        vmovdqu64  zmm5,ZMMWORD PTR [rdx+r10]
        vmovdqu64  zmm6,ZMMWORD PTR [rdx+r10*2]
        vmovdqu64  zmm7,ZMMWORD PTR [rdx+r13]
        vmovdqu64  ZMMWORD PTR [rcx],zmm4
        vmovdqu64  ZMMWORD PTR [rcx+r12],zmm5
        vmovdqu64  ZMMWORD PTR [rcx+r12*2],zmm6
        vmovdqu64  ZMMWORD PTR [rcx+rax],zmm7
        vpmaddubsw zmm4,zmm4,zmm9               # horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm4               # add words to row accumulators
        vpmaddubsw zmm5,zmm5,zmm9
        vpaddw     zmm1,zmm1,zmm5
        vpmaddubsw zmm6,zmm6,zmm9
        vpaddw     zmm2,zmm2,zmm6
        vpmaddubsw zmm7,zmm7,zmm9
        vpaddw     zmm3,zmm3,zmm7
        add     rdx,64                          # src += 64
        add     rcx,64                          # dst += 64
        sub     rbx,64                          # k -= 64
        jae     .LCopyPackA.ProcessNextColumnLoopM4

.LCopyPackA.ProcessRemainingColumnsM4:
        add     rbx,64                          # correct for over-subtract above
        jz      .LCopyPackA.ReduceRowSumBufferM4
        vmovdqu8   zmm4{k1}{z},ZMMWORD PTR [rdx]
        vmovdqu8   zmm5{k1}{z},ZMMWORD PTR [rdx+r10]
        vmovdqu8   zmm6{k1}{z},ZMMWORD PTR [rdx+r10*2]
        vmovdqu8   zmm7{k1}{z},ZMMWORD PTR [rdx+r13]
        vmovdqu64  ZMMWORD PTR [rcx],zmm4
        vmovdqu64  ZMMWORD PTR [rcx+r12],zmm5
        vmovdqu64  ZMMWORD PTR [rcx+r12*2],zmm6
        vmovdqu64  ZMMWORD PTR [rcx+rax],zmm7
        vpmaddubsw zmm4,zmm4,zmm9               # horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm4               # add words to row accumulators
        vpmaddubsw zmm5,zmm5,zmm9
        vpaddw     zmm1,zmm1,zmm5
        vpmaddubsw zmm6,zmm6,zmm9
        vpaddw     zmm2,zmm2,zmm6
        vpmaddubsw zmm7,zmm7,zmm9
        vpaddw     zmm3,zmm3,zmm7

//
// Reduce the sums for the four rows of output.
//

.LCopyPackA.ReduceRowSumBufferM4:
        vpmaddwd       zmm0,zmm0,zmm8           # horizontal word+word=dword per row
        vpmaddwd       zmm1,zmm1,zmm8
        vpmaddwd       zmm2,zmm2,zmm8
        vpmaddwd       zmm3,zmm3,zmm8
        vextracti64x4  ymm4,zmm0,1              # fold zmm -> ymm
        vextracti64x4  ymm5,zmm1,1
        vextracti64x4  ymm6,zmm2,1
        vextracti64x4  ymm7,zmm3,1
        vpaddd         ymm0,ymm0,ymm4
        vpaddd         ymm1,ymm1,ymm5
        vpaddd         ymm2,ymm2,ymm6
        vpaddd         ymm3,ymm3,ymm7
        vphaddd        ymm0,ymm0,ymm1           # reduce and interleave Sum1/Sum0
        vphaddd        ymm1,ymm2,ymm3           # reduce and interleave Sum3/Sum2
        vphaddd        ymm0,ymm0,ymm1           # reduce and interleave Sum3/Sum2/Sum1/Sum0
        vextracti128   xmm1,ymm0,1              # fold ymm -> xmm
        vpaddd         xmm0,xmm0,xmm1
        vmovdqu        XMMWORD PTR [r9],xmm0
        add     r9,4*4                          # advance row sum buffer by 4 dwords
        sub     r11,4                           # m -= 4
        jae     .LCopyPackA.ProcessNextRowM4

.LCopyPackA.ProcessRemainingRows:
        add     r11,4                       # correct for over-subtract above
        jz      .LCopyPackA.ExitRoutine

//
// Process a single row of matrix A in a loop.
//

.LCopyPackA.ProcessNextRowM1:
        vpxor   xmm0,xmm0,xmm0                  # clear row accumulator
        mov     rdx,rsi                         # src = A
        mov     rcx,rdi                         # dst = D
        add     rsi,r10                         # A to next row
        add     rdi,r12                         # D to next row
        mov     rbx,r8                          # k = CountK
        sub     rbx,64                          # k -= 64
        jb      .LCopyPackA.ProcessRemainingColumnsM1

.LCopyPackA.ProcessNextColumnLoopM1:
        vmovdqu64  zmm4,ZMMWORD PTR [rdx]
        vmovdqu64  ZMMWORD PTR [rcx],zmm4
        vpmaddubsw zmm4,zmm4,zmm9           # horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm4           # add words to row accumulators
        add     rdx,64                      # src += 64
        add     rcx,64                      # dst += 64
        sub     rbx,64                      # k -= 64
        jae     .LCopyPackA.ProcessNextColumnLoopM1

.LCopyPackA.ProcessRemainingColumnsM1:
        add     rbx,64                      # correct for over-subtract above
        jz      .LCopyPackA.ReduceRowSumBufferM1
        vmovdqu8   zmm4{k1}{z},ZMMWORD PTR [rdx]
        vmovdqu64  ZMMWORD PTR [rcx],zmm4
        vpmaddubsw zmm4,zmm4,zmm9           # horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm4           # add words to row accumulators

//
// Reduce the sum for the single row of output.
//

.LCopyPackA.ReduceRowSumBufferM1:
        vpmaddwd       zmm0,zmm0,zmm8           # horizontal word+word=dword per row
        vextracti64x4  ymm4,zmm0,1             # fold zmm -> ymm
        vpaddd         ymm0,ymm0,ymm4
        vextracti128   xmm1,ymm0,1              # fold ymm -> xmm
        vpaddd         xmm0,xmm0,xmm1           # reduction
        vphaddd        xmm0,xmm0,xmm0
        vphaddd        xmm0,xmm0,xmm0
        vmovd          DWORD PTR [r9],xmm0
        add     r9,4                        # advance row sum buffer by 1 dword
        dec     r11                         # decrement rows remaining
        jnz     .LCopyPackA.ProcessNextRowM1

//
// Restore non-volatile registers and return.
//

.LCopyPackA.ExitRoutine:
        vzeroupper

        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret


//
// Stack frame layout for the U8X8 kernel.
//

        .equ    .LGemmU8X8KernelFrame_C, -40
        .equ    .LGemmU8X8KernelFrame_B, -32
        .equ    .LGemmU8X8KernelFrame_CountN, -24
        .equ    .LGemmU8X8KernelFrame_CountM, -16
        .equ    .LGemmU8X8KernelFrame_padding, -8
        .equ    .LGemmU8X8KernelFrame_SavedR15, 0
        .equ    .LGemmU8X8KernelFrame_SavedR14, 8
        .equ    .LGemmU8X8KernelFrame_SavedR13, 16
        .equ    .LGemmU8X8KernelFrame_SavedR12, 24
        .equ    .LGemmU8X8KernelFrame_SavedRbx, 32
        .equ    .LGemmU8X8KernelFrame_SavedRbp, 40
        .equ    .LGemmU8X8KernelFrame_ReturnAddress, 48
        .equ    .LGemmU8X8KernelFrame_ldc, 56
        .equ    .LGemmU8X8KernelFrame_RowSumBuffer, 64
        .equ    .LGemmU8X8KernelFrame_ColumnSumBuffer, 72
        .equ    .LGemmU8X8KernelFrame_ZeroPointB, 80
        .equ    .LGemmU8X8KernelFrame_ZeroMode, 88


/*++
Routine Description:
    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.
Arguments:
    A (rdi) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8X8CopyPackAAvx2.
    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8X8CopyPackBAvx2.
    C (rdx) - Supplies the address of matrix C.
    PackedCountK (rcx) - Supplies the number of packed columns from matrix A and
        the number of packed rows from matrix B to iterate over.
    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.
    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.
    ldc - Supplies the first dimension of matrix C.
    RowSumBuffer - Supplies the sum of each row from matrix A. These values have
        been pre-scaled by the zero point offset of matrix B if the offset is
        per-tensor (ZeroPointB is nullptr). Otherwise, these values must be
        scaled by the per-column zero point offsets of matrix B. These values are
        accumulated into every row of matrix C.
    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.
    ZeroPointB - Optionally supplies the per-column zero point offsets of matrix
        B, else nullptr if the matrix B is using per-tensor quantization.
    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.
Return Value:
    Returns the number of rows handled.
--*/

        FUNCTION_ENTRY MlasGemmU8X8KernelAmx

        push    rbp
        push    rbx
        push    r12
        push    r13
        push    r14
        push    r15
 //       mov     r11,.LGemmU8X8KernelFrame_RowSumBuffer[rsp]
 //       mov     r12,.LGemmU8X8KernelFrame_ColumnSumBuffer[rsp]
 //       mov     r13,.LGemmU8X8KernelFrame_ZeroPointB[rsp]
        mov     rax,.LGemmU8X8KernelFrame_ldc[rsp]
        movzx   r10,BYTE PTR .LGemmU8X8KernelFrame_ZeroMode[rsp]
        and     r8,NOT 31                                       # only process 32x rows
        mov     QWORD PTR .LGemmU8X8KernelFrame_CountM[rsp],r8  # Save CountM
        mov     QWORD PTR .LGemmU8X8KernelFrame_CountN[rsp],r9  # Save CountN
        mov     QWORD PTR .LGemmU8X8KernelFrame_B[rsp],rsi      # Save B
        mov     QWORD PTR .LGemmU8X8KernelFrame_C[rsp],rdx      # Save C
        shl     rcx,6                           # PackedCountK -> CountK
        shl     rax,2                           # convert ldc to bytes
        mov     r14,rcx
        shl     r14,4                           # CountK * 16

.LGemmU8S8Kernel.RowM32Loop:
        mov     r9,QWORD PTR .LGemmU8X8KernelFrame_CountN[rsp]  # n = CountN
        mov     rbx,QWORD PTR .LGemmU8X8KernelFrame_C[rsp]  # c_blk = C
        shl     rax,4                           # TILE_M * ldc
        lea     r12,[rbx+rax*2]
        mov     QWORD PTR .LGemmU8X8KernelFrame_C[rsp],r12  # C += 2 * TILE_M * ldc
        mov     rsi,QWORD PTR .LGemmU8X8KernelFrame_B[rsp]      # b_blk = B
        lea     r15,[rbx+rax]                   # c16_blk = c_blk + ldc * TILE_M
        shr     rax,4                           # restore ldc
        sub     r9,32                           # n -= 2 * TILE_N
//        jb      .LGemmU8S8Kernel.ProcessRemainingColumnsM32

.LGemmU8S8Kernel.ColumnM32N32Loop:
        test    r10b, r10b                      # ZeroMode
        je      .LGemmU8S8Kernel.InitAccTileWithC

        tilezero        tmm4
        tilezero        tmm5
        tilezero        tmm6
        tilezero        tmm7

 .LGemmU8S8Kernel.BlockM32N32K64Prefix:
        mov             r13,rdi                 # a_blk = A;
        mov             r12,64                  # TILE_K
        lea             r11,[rsi+r14]           # b16_blk = b_blk + CountK * TILE_N
        add             r14,r13                 # a16_blk = a_blk + CountK * TILE_M
        tileloadd       tmm2,[rdi+rcx*1]        # a_blk, CountK
        tileloadd       tmm0,[rsi+r12*1]        # b_blk, TILE_K
        tileloadd       tmm3,[r14+rcx*1]        # a16_blk, CountK
        tileloadd       tmm1,[r11+r12*1]        # b16_blk, TILE_K
        mov             r15,rcx                 # k = CountK
        sub             r15,64                  # k -= TILE_K
        jz      .LGemmU8S8Kernel.BlockM32N32K64Postfix

.LGemmU8S8Kernel.BlockM32N32K64Loop:
        tdpbusd         tmm4,tmm2,tmm0
        add             rsi,16*64               # b_blk += TILE_N * TILE_K
        add             r13,64                  # a_blk += TILE_K;
        add             r11,16*64               # b16_blk += TILE_N * TILE_K
        add             r14,64                  # a16_blk += TILE_K;
        tdpbusd         tmm5,tmm3,tmm0
        tileloadd       tmm0,[rsi+r12*1]        # b_blk, TILE_K
        tdpbusd         tmm6,tmm2,tmm1
        tileloadd       tmm2,[r13+rcx*1]        # a_blk, CountK
        tdpbusd         tmm7,tmm3,tmm1
        tileloadd       tmm3,[r14+rcx*1]        # a16_blk, CountK
        tileloadd       tmm1,[r11+r12*1]        # b16_blk, TILE_K
        sub             r15,64                  # k -= TILE_K
        ja      .LGemmU8S8Kernel.BlockM32N32K64Loop

.LGemmU8S8Kernel.BlockM32N32K64Postfix:
        tdpbusd         tmm4,tmm2,tmm0
        mov             r15,rax
        shl             r15,4                   # ldc * TILE_M
        add             r15,rbx                 # c16_blk = c_blk + ldc * TILE_M
        tdpbusd         tmm5,tmm3,tmm0
        mov             r14,rcx
        shl             r14,4                   # CountK * 16
        lea             rsi,[rsi+r14+16*64]     # b_blk += CountK * TILE_N + TILE_N * TILE_K
        tdpbusd         tmm6,tmm2,tmm1
        tdpbusd         tmm7,tmm3,tmm1
        tilestored      [rbx+rax*1],tmm4        # c_blk, ldc
        tilestored      [r15+rax*1],tmm5        # c16_blk, ldc
        tilestored      64[rbx+rax*1],tmm6      # c_blk + TILE_N, ldc
        tilestored      64[r15+rax*1],tmm7      # c16_blk + TILE_N, ldc
        add             rbx,4*2*16              # c_blk += 2 * TILE_N
        add             r15,4*2*16
        sub             r9,32                   # n -= 32
        jae     .LGemmU8S8Kernel.ColumnM32N32Loop

//.LGemmU8S8Kernel.ProcessRemainingColumnsM32:
//        add     r9,32                           # correct for over-subtract above
//        jz      .LGemmU8S8Kernel.ReduceRowSumBufferM32

        // process remaining less than 32 columns


//.LGemmU8S8Kernel.ReduceRowSumBufferM32:

        // Go on to next block of rows
//        RowSumBuffer += 2 * TILE_M;
        lea     rdi,[rdi+r14*2]                 # A += 2 * TILE_M * CountK
        sub     r8,32                           # m -= 2 * TILE_M
        ja      .LGemmU8S8Kernel.RowM32Loop

.LExitKernel:
        mov     rax, QWORD PTR .LGemmU8X8KernelFrame_CountM[rsp]
        vzeroupper
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

.LGemmU8S8Kernel.InitAccTileWithC:
        tileloadd       tmm4,[rbx+rax*1]        # c_blk, ldc
        tileloadd       tmm5,[r15+rax*1]        # c16_blk, ldc
        tileloadd       tmm6,64[rbx+rax*1]      # c_blk + TILE_N, ldc
        tileloadd       tmm7,64[r15+rax*1]      # c16_blk + TILE_N, ldc
        jmp	.LGemmU8S8Kernel.BlockM32N32K64Prefix

        .end
