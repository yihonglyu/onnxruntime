/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DepthwiseConvSymKernelNeon.asm

Abstract:

    This module implements the kernels for the depthwise convolution
    operation with symmetrically quantized integer values

--*/

#include "asmmacro.h"

//
// Stack frame layout for the depthwise conv kernel.
// d8-d15, x19-x30 need to be preserved if used
//

        .equ    .LConvSymDepthwiseKernelFrame_SavedNeonRegisters,   (8 * 8)
        .equ    .LConvSymDepthwiseKernelFrame_SavedRegisters,       16 + .LConvSymDepthwiseKernelFrame_SavedNeonRegisters
        .equ    .LConvSymDepthwiseKernelFrame_PostProcessParams,    0 + .LConvSymDepthwiseKernelFrame_SavedRegisters
        .equ    .LConvSymDepthwiseKernelFrame_KernelFlags,          8 + .LConvSymDepthwiseKernelFrame_SavedRegisters

        .equ    .LConvSymDepthwisePostProcessParams_Bias,           0
        .equ    .LConvSymDepthwisePostProcessParams_Scale,          8
        .equ    .LConvSymDepthwisePostProcessParams_Min,            16
        .equ    .LConvSymDepthwisePostProcessParams_Max,            20
        .equ    .LConvSymDepthwisePostProcessParams_ZeroPoint,      24

        .equ    MLAS_CONV_SYM_FLAG_INPUT_DIRECT,                    1
        .equ    MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE,               2

        .text

/*++

Routine Description:

    This routine is the inner kernel to compute a depthwise convolution for the
    elements of an output row for a set of filter rows.

Arguments:

    Input (x0) - Supplies the address of the indirection buffer.
 
    Filter (x1) - Supplies the address of the filter buffer.

    Output (x2) - Supplies the address of the output buffer.

    KernelSize (x3) - Supplies the size of the kernel.
 
    Channels (x4) - Supplies the number of input and output channels.
 
    ChannelOffset (x5) - Supplies the byte offset from the indirection buffer base
        address for this iteration.
 
    ChannelCount (x6) - Supplies the number of channels this iteration produces.
 
        This implementation requires the count to be 16 or 8
 
    OutputCount (x7)- Supplies the number of output elements this iteration produces.
 
        This implementation requires the count to be in the range 1 to 2.
 
    PostProcessParams - Supplies the address of the post process parameter block.
 
    KernelFlags - Supplies additional flags controlling the operation.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvSymDepthwiseKernelNeon

        stp     d8,d9,[sp,#-80]!
        ldr     x8,[sp,#.LConvSymDepthwiseKernelFrame_PostProcessParams]
        mov     w10,#0x80808080
        stp     d10,d11,[sp,#16]
        stp     d12,d13,[sp,#32]
        stp     d14,d15,[sp,#48]
        stp     x19,x20,[sp,#64]
        dup     v8.4s,w10                   // bit flip vector
        ldr     x16,[x8,#.LConvSymDepthwisePostProcessParams_Bias]
        cmp     x7,2
        add     x9,x0,x3,lsl#3              // x9 -> &A1
        add     x14,x0,x3,lsl#4             // x14 -> &A2
        add     x15,x9,x3,lsl#4             // x15 -> &A3
        csel    x9,x0,x9,lo                 // x9 -> &A0 if OutputCount < 2
        csel    x14,x0,x14,ls               // x14 -> &A0 if OutputCount <= 2
        ldr     x11,[x9],#8                 // x11 -> A1 iter 0
        cmp     x7,4
        ldp     q24,q25,[x16],#32           // init accumulators with bias
        csel    x15,x0,x15,lo               // x15 -> &A0 if OutputCount < 4
        cmp     x6,16
        ldr     x10,[x0],#8                 // x10 -> A0 iter 0
        b.lo    .LProcess8Channels

//
// Process an input block of length Channels for each element of the kernel.
//
// Filter:  x16|x17 - v0|v1
//          x19|x20 - v0|v1  // unroll
// Input:
// x0  -> x10 -> v4|v5
//     -> x12 -> v2|v3   // unroll
// x9  -> x11 -> v6|v7
//     -> x13 -> v10|v11 // unroll
// x14 -> x10 -> v4|v5
//     -> x12 -> v2|v3   // unroll
// x15 -> x11 -> v6|v7
//     -> x13 -> v10|v11 // unroll
//

.LProcess16Channels:
        cmp     x3,1
        add     x6,x5,#8
        ldp     q26,q27,[x16]
        b.eq    .LProcC16P1

        mov     v28.16b,v24.16b
        ldr     x12,[x0],#8                 // x12 -> A0 iter 1
        mov     v29.16b,v25.16b
        ldr     x13,[x9],#8                 // x13 -> A1 iter 1
        ldr     x16,[x1]                    // filter L iter 0
        ldr     x17,[x1,#8]                 // filter H iter 0
        mov     v16.16b,v24.16b
        add     x1,x1,x4
        ldr     x19,[x1]                    // filter L iter 1
        ldr     x20,[x1,#8]                 // filter H iter 1
        mov     v17.16b,v25.16b
        add     x1,x1,x4
        ldr     d4,[x10,x5]                 // A0 L iter 0
        ldr     d5,[x10,x6]                 // A0 H iter 0
        mov     v20.16b,v24.16b
        ldr     x10,[x14],#8                // x10 -> A2 iter 0
        mov     v21.16b,v25.16b
        ldr     d6,[x11,x5]                 // A1 L iter 0
        ldr     d7,[x11,x6]                 // A1 H iter 0
        mov     v30.16b,v26.16b
        ldr     x11,[x15],#8                // x11 -> A3 iter 0
        mov     v31.16b,v27.16b
        ldr     d2,[x12,x5]                 // A0 L iter 1
        ldr     d3,[x12,x6]                 // A0 H iter 1
        subs    x3,x3,2                     // decrement input blocks remaining
        mov     v18.16b,v26.16b
        ldr     x12,[x14],#8                // x12 -> A2 iter 1
        mov     v19.16b,v27.16b
        ldr     d10,[x13,x5]                // A1 L iter 1
        ldr     d11,[x13,x6]                // A1 H iter 1
        mov     v22.16b,v26.16b
        ldr     x13,[x15],#8                // x13 -> A3 iter 1
        mov     v23.16b,v27.16b
        ins     v0.d[0],x16
        ins     v1.d[0],x17
        eor     v4.8b,v4.8b,v8.8b           // fix sign bits
        eor     v5.8b,v5.8b,v8.8b

.LBlockLoopC16:

        //
        // Process 2 pixels, and load next two pixels
        //
        smull   v12.8h,v0.8b,v4.8b
        ldr     d4,[x10,x5]                 // A2 L iter 0
        smull   v13.8h,v1.8b,v5.8b
        ldr     d5,[x10,x6]                 // A2 H iter 0
        eor     v6.8b,v6.8b,v8.8b
        eor     v7.8b,v7.8b,v8.8b
        smull   v14.8h,v0.8b,v6.8b
        ldr     d6,[x11,x5]                 // A3 L iter 0
        smull   v15.8h,v1.8b,v7.8b
        ldr     d7,[x11,x6]                 // A3 H iter 0
        ins     v0.d[0],x19
        ins     v1.d[0],x20
        eor     v2.8b,v2.8b,v8.8b
        eor     v3.8b,v3.8b,v8.8b
        smlal   v12.8h,v0.8b,v2.8b
        ldr     d2,[x12,x5]                 // A2 L iter 1
        smlal   v13.8h,v1.8b,v3.8b
        ldr     d3,[x12,x6]                 // A2 H iter 1
        eor     v10.8b,v10.8b,v8.8b
        eor     v11.8b,v11.8b,v8.8b
        b.eq    .LEpilogueC16P2
        smlal   v14.8h,v0.8b,v10.8b
        ldr     d10,[x13,x5]                // A3 L iter 1
        smlal   v15.8h,v1.8b,v11.8b
        cmp     x3,1
        ldr     d11,[x13,x6]                // A3 H iter 1
        saddw   v24.4s,v24.4s,v12.4h
        ldr     x10,[x0],#8                 // x10 -> A0 iter 2
        saddw2  v25.4s,v25.4s,v12.8h
        ldr     x11,[x9],#8                 // x11 -> A1 iter 2
        saddw   v26.4s,v26.4s,v13.4h
        b.eq    .LEpilogueC16P3             // 3 pixel remains

        ldr     x12,[x0],#8                 // x12 -> A0 iter 3
        saddw2  v27.4s,v27.4s,v13.8h
        ldr     x13,[x9],#8                 // x13 -> A1 iter 3
        saddw   v28.4s,v28.4s,v14.4h
        ins     v0.d[0],x16
        saddw2  v29.4s,v29.4s,v14.8h
        ins     v1.d[0],x17
        saddw   v30.4s,v30.4s,v15.4h
        eor     v4.8b,v4.8b,v8.8b
        saddw2  v31.4s,v31.4s,v15.8h
        eor     v5.8b,v5.8b,v8.8b
        smull   v12.8h,v0.8b,v4.8b
        ldr     d4,[x10,x5]                 // A0 L iter 2
        smull   v13.8h,v1.8b,v5.8b
        ldr     d5,[x10,x6]                 // A0 H iter 2
        eor     v6.8b,v6.8b,v8.8b
        eor     v7.8b,v7.8b,v8.8b
        smull   v14.8h,v0.8b,v6.8b
        ldr     d6,[x11,x5]                 // A1 L iter 2
        smull   v15.8h,v1.8b,v7.8b
        ldr     d7,[x11,x6]                 // A1 H iter 2
        ins     v0.d[0],x19
        ins     v1.d[0],x20
        eor     v2.8b,v2.8b,v8.8b
        eor     v3.8b,v3.8b,v8.8b
        ldr     x16,[x1]                    // filter L iter 2
        ldr     x17,[x1,#8]                 // filter H iter 2
        add     x1,x1,x4
        subs    x3,x3,2                     // decrement input blocks remaining
        smlal   v12.8h,v0.8b,v2.8b
        ldr     d2,[x12,x5]                 // A0 L iter 3
        smlal   v13.8h,v1.8b,v3.8b
        ldr     d3,[x12,x6]                 // A0 H iter 3
        eor     v10.8b,v10.8b,v8.8b
        eor     v11.8b,v11.8b,v8.8b
        smlal   v14.8h,v0.8b,v10.8b
        ldr     d10,[x13,x5]                // A1 L iter 3
        smlal   v15.8h,v1.8b,v11.8b
        ldr     d11,[x13,x6]                // A1 H iter 3
        ldr     x19,[x1]                    // filter L iter 3
        ldr     x20,[x1,#8]                 // filter H iter 3
        add     x1,x1,x4
        ldr     x10,[x14],#8                // x10 -> A2 iter 2
        saddw   v16.4s,v16.4s,v12.4h
        ldr     x11,[x15],#8                // x11 -> A3 iter 2
        saddw2  v17.4s,v17.4s,v12.8h
        ldr     x12,[x14],#8                // x12 -> A2 iter 3
        saddw   v18.4s,v18.4s,v13.4h
        ldr     x13,[x15],#8                // x13 -> A3 iter 3
        saddw2  v19.4s,v19.4s,v13.8h
        ins     v0.d[0],x16
        saddw   v20.4s,v20.4s,v14.4h
        ins     v1.d[0],x17
        saddw2  v21.4s,v21.4s,v14.8h
        eor     v4.8b,v4.8b,v8.8b           // fix sign bits
        saddw   v22.4s,v22.4s,v15.4h
        eor     v5.8b,v5.8b,v8.8b
        saddw2  v23.4s,v23.4s,v15.8h
        b       .LBlockLoopC16

.LEpilogueC16P2:
        //
        // Loop epilogue (process last 2 pixels) mixed
        // with loading of dequantization params
        //
        smlal   v14.8h,v0.8b,v10.8b
        ldr     d10,[x13,x5]                // A3 L iter 1
        smlal   v15.8h,v1.8b,v11.8b
        ldr     d11,[x13,x6]                // A3 H iter 1
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        saddw   v26.4s,v26.4s,v13.4h
        saddw2  v27.4s,v27.4s,v13.8h
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        saddw   v30.4s,v30.4s,v15.4h
        saddw2  v31.4s,v31.4s,v15.8h
        ldr     w9,[sp,#.LConvSymDepthwiseKernelFrame_KernelFlags]
        eor     v4.8b,v4.8b,v8.8b
        eor     v5.8b,v5.8b,v8.8b
        ldr     x12,[x8,#.LConvSymDepthwisePostProcessParams_Scale]
        ins     v0.d[0],x16
        ins     v1.d[0],x17
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        eor     v6.8b,v6.8b,v8.8b
        eor     v7.8b,v7.8b,v8.8b
        ldr     w15,[x8,#.LConvSymDepthwisePostProcessParams_ZeroPoint]
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        eor     v2.8b,v2.8b,v8.8b
        eor     v3.8b,v3.8b,v8.8b
        ins     v0.d[0],x19
        ins     v1.d[0],x20
        smlal   v12.8h,v0.8b,v2.8b
        smlal   v13.8h,v1.8b,v3.8b
        eor     v10.8b,v10.8b,v8.8b
        eor     v11.8b,v11.8b,v8.8b
        smlal   v14.8h,v0.8b,v10.8b
        smlal   v15.8h,v1.8b,v11.8b
        tst     w9,#MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        ld1r    {v4.4s},[x12]               // load scale val
        b.eq    .LSkipScaleVecLoad2
        ldp     q4,q11,[x12],#32            // load scale vector if per channel
        ldp     q6,q9,[x12]
.LSkipScaleVecLoad2:
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v18.4s,v18.4s,v13.4h
        saddw2  v19.4s,v19.4s,v13.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h
        saddw   v22.4s,v22.4s,v15.4h
        saddw2  v23.4s,v23.4s,v15.8h
        b       .LDequantization

.LProcC16P1:
        //
        // Channel 16 kernel size 1
        // TODO!! seperate kernel for this?
        //
        ldr     x12,[x14],#8                // x12 -> A2
        ldr     x13,[x15],#8                // x13 -> A3
        mov     v28.16b,v24.16b
        mov     v29.16b,v25.16b
        ldr     d0,[x1]
        ldr     d1,[x1,#8]
        mov     v16.16b,v24.16b
        mov     v17.16b,v25.16b
        ldr     d4,[x10,x5]
        ldr     d5,[x10,x6]
        mov     v20.16b,v24.16b
        mov     v21.16b,v25.16b
        ldr     d6,[x11,x5]
        ldr     d7,[x11,x6]
        mov     v30.16b,v26.16b
        mov     v31.16b,v27.16b
        ldr     d2,[x12,x5]
        ldr     d3,[x12,x6]
        subs    x3,x3,2                     // decrement input blocks remaining
        mov     v18.16b,v26.16b
        mov     v19.16b,v27.16b
        ldr     d10,[x13,x5]
        ldr     d11,[x13,x6]
        mov     v22.16b,v26.16b
        mov     v23.16b,v27.16b
        b       .LEpilogueC16P1

.LEpilogueC16P3:
        //
        // Loop epilogue (process last 2 pixels) mixed
        // with loading of dequantization params
        //
        ldr     x12,[x14],#8                // x12 -> A2 iter 2
        saddw2  v27.4s,v27.4s,v13.8h
        ldr     x13,[x15],#8                // x13 -> A3 iter 2
        saddw   v28.4s,v28.4s,v14.4h
        ins     v0.d[0],x16
        saddw2  v29.4s,v29.4s,v14.8h
        ins     v1.d[0],x17
        saddw   v30.4s,v30.4s,v15.4h
        eor     v4.8b,v4.8b,v8.8b
        saddw2  v31.4s,v31.4s,v15.8h
        eor     v5.8b,v5.8b,v8.8b
        smull   v12.8h,v0.8b,v4.8b
        ldr     d4,[x10,x5]                 // A0 L iter 2
        smull   v13.8h,v1.8b,v5.8b
        ldr     d5,[x10,x6]                 // A0 H iter 2
        eor     v6.8b,v6.8b,v8.8b
        eor     v7.8b,v7.8b,v8.8b
        smull   v14.8h,v0.8b,v6.8b
        ldr     d6,[x11,x5]                 // A1 L iter 2
        smull   v15.8h,v1.8b,v7.8b
        ldr     d7,[x11,x6]                 // A1 H iter 2
        ins     v0.d[0],x19
        ins     v1.d[0],x20
        eor     v2.8b,v2.8b,v8.8b
        eor     v3.8b,v3.8b,v8.8b
        smlal   v12.8h,v0.8b,v2.8b
        ldr     d2,[x12,x5]                 // A2 L iter 2
        smlal   v13.8h,v1.8b,v3.8b
        ldr     d3,[x12,x6]                 // A2 H iter 2
        eor     v10.8b,v10.8b,v8.8b
        eor     v11.8b,v11.8b,v8.8b
        smlal   v14.8h,v0.8b,v10.8b
        ldr     d10,[x13,x5]                // A3 L iter 2
        smlal   v15.8h,v1.8b,v11.8b
        ldr     d11,[x13,x6]                // A3 H iter 2
        ldr     d0,[x1]                     // filter L iter 2
        ldr     d1,[x1,#8]                  // filter H iter 2
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v18.4s,v18.4s,v13.4h
        saddw2  v19.4s,v19.4s,v13.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h
        saddw   v22.4s,v22.4s,v15.4h
        saddw2  v23.4s,v23.4s,v15.8h

.LEpilogueC16P1:
        //
        // Loop epilogue (process last single pixel) mixed with loading of dequantization params
        //
        ldr     w9,[sp,#.LConvSymDepthwiseKernelFrame_KernelFlags]
        eor     v4.8b,v4.8b,v8.8b
        eor     v5.8b,v5.8b,v8.8b
        ldr     x12,[x8,#.LConvSymDepthwisePostProcessParams_Scale]
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        eor     v6.8b,v6.8b,v8.8b
        eor     v7.8b,v7.8b,v8.8b
        ldr     w15,[x8,#.LConvSymDepthwisePostProcessParams_ZeroPoint]
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        saddw   v26.4s,v26.4s,v13.4h
        saddw2  v27.4s,v27.4s,v13.8h
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        saddw   v30.4s,v30.4s,v15.4h
        saddw2  v31.4s,v31.4s,v15.8h
        eor     v2.8b,v2.8b,v8.8b
        eor     v3.8b,v3.8b,v8.8b
        smull   v12.8h,v0.8b,v2.8b
        smull   v13.8h,v1.8b,v3.8b
        eor     v10.8b,v10.8b,v8.8b
        eor     v11.8b,v11.8b,v8.8b
        smull   v14.8h,v0.8b,v10.8b
        smull   v15.8h,v1.8b,v11.8b
        tst     w9,#MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        ld1r    {v4.4s},[x12]               // load scale val
        b.eq    .LSkipScaleVecLoad
        ldp     q4,q11,[x12],#32            // load scale vector if per channel
        ldp     q6,q9,[x12]
.LSkipScaleVecLoad:
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v18.4s,v18.4s,v13.4h
        saddw2  v19.4s,v19.4s,v13.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h
        saddw   v22.4s,v22.4s,v15.4h
        saddw2  v23.4s,v23.4s,v15.8h

.LDequantization:
        scvtf   v24.4s,v24.4s               // convert to float
        scvtf   v25.4s,v25.4s
        scvtf   v26.4s,v26.4s
        scvtf   v27.4s,v27.4s
        scvtf   v28.4s,v28.4s
        scvtf   v29.4s,v29.4s
        scvtf   v30.4s,v30.4s
        scvtf   v31.4s,v31.4s
        scvtf   v16.4s,v16.4s
        scvtf   v17.4s,v17.4s
        scvtf   v18.4s,v18.4s
        scvtf   v19.4s,v19.4s
        scvtf   v20.4s,v20.4s
        scvtf   v21.4s,v21.4s
        scvtf   v22.4s,v22.4s
        scvtf   v23.4s,v23.4s
        b.ne    .LSkipScaleBroadcast
        mov     v11.16b,v4.16b               // broadcast scale val if not per channel
        mov     v6.16b,v4.16b
        mov     v9.16b,v4.16b
.LSkipScaleBroadcast:
        fmul    v24.4s,v24.4s,v4.4s         // multiply by scale
        fmul    v25.4s,v25.4s,v11.4s
        fmul    v26.4s,v26.4s,v6.4s
        fmul    v27.4s,v27.4s,v9.4s
        fmul    v28.4s,v28.4s,v4.4s
        fmul    v29.4s,v29.4s,v11.4s
        fmul    v30.4s,v30.4s,v6.4s
        fmul    v31.4s,v31.4s,v9.4s
        fmul    v16.4s,v16.4s,v4.4s
        fmul    v17.4s,v17.4s,v11.4s
        fmul    v18.4s,v18.4s,v6.4s
        fmul    v19.4s,v19.4s,v9.4s
        fmul    v20.4s,v20.4s,v4.4s
        fmul    v21.4s,v21.4s,v11.4s
        fmul    v22.4s,v22.4s,v6.4s
        fmul    v23.4s,v23.4s,v9.4s
        fcvtns  v24.4s,v24.4s               // convert to int
        fcvtns  v25.4s,v25.4s
        fcvtns  v26.4s,v26.4s
        fcvtns  v27.4s,v27.4s
        fcvtns  v28.4s,v28.4s
        fcvtns  v29.4s,v29.4s
        fcvtns  v30.4s,v30.4s
        fcvtns  v31.4s,v31.4s
        fcvtns  v16.4s,v16.4s
        fcvtns  v17.4s,v17.4s
        fcvtns  v18.4s,v18.4s
        fcvtns  v19.4s,v19.4s
        fcvtns  v20.4s,v20.4s
        fcvtns  v21.4s,v21.4s
        fcvtns  v22.4s,v22.4s
        fcvtns  v23.4s,v23.4s
        sqxtn   v24.4h,v24.4s               // shorten to int16
        sqxtn   v26.4h,v26.4s
        sqxtn2  v24.8h,v25.4s
        sqxtn2  v26.8h,v27.4s
        sqxtn   v28.4h,v28.4s
        sqxtn   v30.4h,v30.4s
        sqxtn2  v28.8h,v29.4s
        sqxtn2  v30.8h,v31.4s
        dup     v0.8h,w15
        sqxtn   v16.4h,v16.4s
        sqxtn   v18.4h,v18.4s
        sqxtn2  v16.8h,v17.4s
        sqxtn2  v18.8h,v19.4s
        sqxtn   v20.4h,v20.4s
        sqxtn   v22.4h,v22.4s
        sqxtn2  v20.8h,v21.4s
        sqxtn2  v22.8h,v23.4s
        sqadd   v24.8h,v24.8h,v0.8h         // add zero point
        sqadd   v26.8h,v26.8h,v0.8h
        sqadd   v28.8h,v28.8h,v0.8h
        sqadd   v30.8h,v30.8h,v0.8h
        sqadd   v16.8h,v16.8h,v0.8h
        sqadd   v18.8h,v18.8h,v0.8h
        sqadd   v20.8h,v20.8h,v0.8h
        sqadd   v22.8h,v22.8h,v0.8h
        sqxtun  v24.8b,v24.8h               // shorten to int8
        sqxtun2 v24.16b,v26.8h
        sqxtun  v28.8b,v28.8h
        sqxtun2 v28.16b,v30.8h
        sqxtun  v16.8b,v16.8h
        sqxtun2 v16.16b,v18.8h
        sqxtun  v20.8b,v20.8h
        sqxtun2 v20.16b,v22.8h
        cmp     x7,2                        // OutputCount < 2 ?
        st1     {v24.16b},[x2],x4
        b.lo    .LExitKernel                // exit if OutputCount < 2
        st1     {v28.16b},[x2],x4
        b.ls    .LExitKernel                // exit if OutputCount <=2
        cmp     x7,4                        // OutputCount < 4 ?
        st1     {v16.16b},[x2],x4
        b.lo    .LExitKernel                // exit if OutputCount < 4
        str     q20,[x2]


.LExitKernel:
        ldp     x19,x20,[sp,#64]
        ldp     d14,d15,[sp,#48]
        ldp     d12,d13,[sp,#32]
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#80
        ret

//
// Process an input block of length Channels for each element of the kernel.
//
// Filter:  v0
// Input:
// x0  -> x10 -> v4
// x9  -> x11 -> v6
// x14 -> x12 -> v2
// x15 -> x13 -> v10
//
.LProcess8Channels:
        ldr     x12,[x14],#8                // x12 -> A2
        ldr     x13,[x15],#8                // x13 -> A3
        mov     v28.16b,v24.16b
        mov     v29.16b,v25.16b
        mov     v16.16b,v24.16b
        mov     v17.16b,v25.16b
        ld1     {v0.d}[0],[x1],x4
        subs    x3,x3,1                     // decrement input blocks remaining
        ldr     d4,[x10,x5]
        ldr     d6,[x11,x5]
        ldr     d2,[x12,x5]
        ldr     d10,[x13,x5]
        mov     v20.16b,v24.16b
        mov     v21.16b,v25.16b

.LBlockLoopC8:
        b.ls    .LDequantizeC8
        ldr     x10,[x0],#8                 // x10 -> A0
        ldr     x11,[x9],#8                 // x11 -> A1
        ldr     x12,[x14],#8                // x12 -> A2
        ldr     x13,[x15],#8                // x13 -> A3
        eor     v4.8b,v4.8b,v8.8b           // fix sign bits
        eor     v6.8b,v6.8b,v8.8b
        eor     v2.8b,v2.8b,v8.8b
        eor     v10.8b,v10.8b,v8.8b
        smull   v5.8h,v0.8b,v4.8b
        smull   v7.8h,v0.8b,v6.8b
        smull   v3.8h,v0.8b,v2.8b
        smull   v11.8h,v0.8b,v10.8b
        add     x10,x10,x5
        add     x11,x11,x5
        add     x12,x12,x5
        add     x13,x13,x5
        ldr     d4,[x10]
        ld1     {v0.d}[0],[x1],x4
        ldr     d6,[x11]
        ldr     d2,[x12]
        ldr     d10,[x13]
        saddw   v24.4s,v24.4s,v5.4h
        saddw2  v25.4s,v25.4s,v5.8h
        saddw   v28.4s,v28.4s,v7.4h
        saddw2  v29.4s,v29.4s,v7.8h
        subs    x3,x3,1                     // decrement input blocks remaining
        saddw   v16.4s,v16.4s,v3.4h
        saddw2  v17.4s,v17.4s,v3.8h
        saddw   v20.4s,v20.4s,v11.4h
        saddw2  v21.4s,v21.4s,v11.8h
        b       .LBlockLoopC8

.LDequantizeC8:
        ldr     w9,[sp,#.LConvSymDepthwiseKernelFrame_KernelFlags]
        ldr     x12,[x8,#.LConvSymDepthwisePostProcessParams_Scale]
        eor     v4.8b,v4.8b,v8.8b           // fix sign bits
        eor     v6.8b,v6.8b,v8.8b
        eor     v2.8b,v2.8b,v8.8b
        eor     v10.8b,v10.8b,v8.8b
        smull   v5.8h,v0.8b,v4.8b
        smull   v7.8h,v0.8b,v6.8b
        smull   v3.8h,v0.8b,v2.8b
        smull   v11.8h,v0.8b,v10.8b
        ldr     w15,[x8,#.LConvSymDepthwisePostProcessParams_ZeroPoint]
        saddw   v24.4s,v24.4s,v5.4h
        saddw2  v25.4s,v25.4s,v5.8h
        tst     w9,#MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        ld1r    {v4.4s},[x12]               // load scale Value
        b.eq    .LSkipScaleVecLoadC8
        ldp     q4,q5,[x12]                 // load scale vector
.LSkipScaleVecLoadC8:
        saddw   v28.4s,v28.4s,v7.4h
        saddw2  v29.4s,v29.4s,v7.8h
        saddw   v16.4s,v16.4s,v3.4h
        saddw2  v17.4s,v17.4s,v3.8h
        saddw   v20.4s,v20.4s,v11.4h
        saddw2  v21.4s,v21.4s,v11.8h

        scvtf   v24.4s,v24.4s               // convert to float
        scvtf   v25.4s,v25.4s
        scvtf   v28.4s,v28.4s
        scvtf   v29.4s,v29.4s
        scvtf   v16.4s,v16.4s
        scvtf   v17.4s,v17.4s
        scvtf   v20.4s,v20.4s
        scvtf   v21.4s,v21.4s
        b.ne    .LSkipScaleBroadcastC8
        mov     v5.16b,v4.16b
.LSkipScaleBroadcastC8:
        fmul    v24.4s,v24.4s,v4.4s         // multiply by scale
        fmul    v25.4s,v25.4s,v5.4s
        fmul    v28.4s,v28.4s,v4.4s
        fmul    v29.4s,v29.4s,v5.4s
        fmul    v16.4s,v16.4s,v4.4s
        fmul    v17.4s,v17.4s,v5.4s
        fmul    v20.4s,v20.4s,v4.4s
        fmul    v21.4s,v21.4s,v5.4s
        fcvtns  v24.4s,v24.4s               // convert to int
        fcvtns  v25.4s,v25.4s
        fcvtns  v28.4s,v28.4s
        fcvtns  v29.4s,v29.4s
        fcvtns  v16.4s,v16.4s
        fcvtns  v17.4s,v17.4s
        fcvtns  v20.4s,v20.4s
        fcvtns  v21.4s,v21.4s
        dup     v0.8h,w15
        sqxtn   v24.4h,v24.4s               // shorten to int16
        sqxtn2  v24.8h,v25.4s
        sqxtn   v28.4h,v28.4s
        sqxtn2  v28.8h,v29.4s
        sqxtn   v16.4h,v16.4s
        sqxtn2  v16.8h,v17.4s
        sqxtn   v20.4h,v20.4s
        sqxtn2  v20.8h,v21.4s
        sqadd   v24.8h,v24.8h,v0.8h         // add zero point
        sqadd   v28.8h,v28.8h,v0.8h
        sqadd   v16.8h,v16.8h,v0.8h
        sqadd   v20.8h,v20.8h,v0.8h
        sqxtun  v24.8b,v24.8h               // shorten to int8
        sqxtun  v28.8b,v28.8h
        sqxtun  v16.8b,v16.8h
        sqxtun  v20.8b,v20.8h
        cmp     x7,2                        // OutputCount < 2 ?
        st1     {v24.8b},[x2],x4
        b.lo    .LExitKernel                // exit if OutputCount < 2

        st1     {v28.8b},[x2],x4
        b.ls    .LExitKernel                // exit if OutputCount <=2

        cmp     x7,4                        // OutputCount < 4 ?
        st1     {v16.8b},[x2],x4
        b.lo    .LExitKernel                // exit if OutputCount < 4

        st1     {v20.8b},[x2],x4
        b       .LExitKernel

        .end
