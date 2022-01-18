// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// AXILiteS
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of ch_div_K
//        bit 31~0 - ch_div_K[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of height_in
//        bit 31~0 - height_in[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of width_in
//        bit 31~0 - width_in[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of height_out
//        bit 31~0 - height_out[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of width_out
//        bit 31~0 - width_out[31:0] (Read/Write)
// 0x34 : reserved
// 0x38 : Data signal of Kx
//        bit 31~0 - Kx[31:0] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of Ky
//        bit 31~0 - Ky[31:0] (Read/Write)
// 0x44 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XPOOL_AXILITES_ADDR_AP_CTRL         0x00
#define XPOOL_AXILITES_ADDR_GIE             0x04
#define XPOOL_AXILITES_ADDR_IER             0x08
#define XPOOL_AXILITES_ADDR_ISR             0x0c
#define XPOOL_AXILITES_ADDR_CH_DIV_K_DATA   0x10
#define XPOOL_AXILITES_BITS_CH_DIV_K_DATA   32
#define XPOOL_AXILITES_ADDR_HEIGHT_IN_DATA  0x18
#define XPOOL_AXILITES_BITS_HEIGHT_IN_DATA  32
#define XPOOL_AXILITES_ADDR_WIDTH_IN_DATA   0x20
#define XPOOL_AXILITES_BITS_WIDTH_IN_DATA   32
#define XPOOL_AXILITES_ADDR_HEIGHT_OUT_DATA 0x28
#define XPOOL_AXILITES_BITS_HEIGHT_OUT_DATA 32
#define XPOOL_AXILITES_ADDR_WIDTH_OUT_DATA  0x30
#define XPOOL_AXILITES_BITS_WIDTH_OUT_DATA  32
#define XPOOL_AXILITES_ADDR_KX_DATA         0x38
#define XPOOL_AXILITES_BITS_KX_DATA         32
#define XPOOL_AXILITES_ADDR_KY_DATA         0x40
#define XPOOL_AXILITES_BITS_KY_DATA         32

