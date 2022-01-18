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
// 0x10 : Data signal of CHin_V
//        bit 15~0 - CHin_V[15:0] (Read/Write)
//        others   - reserved
// 0x14 : reserved
// 0x18 : Data signal of Hin_V
//        bit 15~0 - Hin_V[15:0] (Read/Write)
//        others   - reserved
// 0x1c : reserved
// 0x20 : Data signal of Win_V
//        bit 15~0 - Win_V[15:0] (Read/Write)
//        others   - reserved
// 0x24 : reserved
// 0x28 : Data signal of CHout_V
//        bit 15~0 - CHout_V[15:0] (Read/Write)
//        others   - reserved
// 0x2c : reserved
// 0x30 : Data signal of Kx_V
//        bit 7~0 - Kx_V[7:0] (Read/Write)
//        others  - reserved
// 0x34 : reserved
// 0x38 : Data signal of Ky_V
//        bit 7~0 - Ky_V[7:0] (Read/Write)
//        others  - reserved
// 0x3c : reserved
// 0x40 : Data signal of Sx_V
//        bit 7~0 - Sx_V[7:0] (Read/Write)
//        others  - reserved
// 0x44 : reserved
// 0x48 : Data signal of Sy_V
//        bit 7~0 - Sy_V[7:0] (Read/Write)
//        others  - reserved
// 0x4c : reserved
// 0x50 : Data signal of mode_V
//        bit 0  - mode_V[0] (Read/Write)
//        others - reserved
// 0x54 : reserved
// 0x58 : Data signal of relu_en_V
//        bit 0  - relu_en_V[0] (Read/Write)
//        others - reserved
// 0x5c : reserved
// 0x60 : Data signal of feature_in_V
//        bit 31~0 - feature_in_V[31:0] (Read/Write)
// 0x64 : reserved
// 0x68 : Data signal of feature_in_precision_V
//        bit 3~0 - feature_in_precision_V[3:0] (Read/Write)
//        others  - reserved
// 0x6c : reserved
// 0x70 : Data signal of W_V
//        bit 31~0 - W_V[31:0] (Read/Write)
// 0x74 : reserved
// 0x78 : Data signal of W_precision_V
//        bit 3~0 - W_precision_V[3:0] (Read/Write)
//        others  - reserved
// 0x7c : reserved
// 0x80 : Data signal of B_V
//        bit 31~0 - B_V[31:0] (Read/Write)
// 0x84 : reserved
// 0x88 : Data signal of B_precision_V
//        bit 3~0 - B_precision_V[3:0] (Read/Write)
//        others  - reserved
// 0x8c : reserved
// 0x90 : Data signal of feature_out_V
//        bit 31~0 - feature_out_V[31:0] (Read/Write)
// 0x94 : reserved
// 0x98 : Data signal of feature_out_precision_V
//        bit 3~0 - feature_out_precision_V[3:0] (Read/Write)
//        others  - reserved
// 0x9c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XCONV_AXILITES_ADDR_AP_CTRL                      0x00
#define XCONV_AXILITES_ADDR_GIE                          0x04
#define XCONV_AXILITES_ADDR_IER                          0x08
#define XCONV_AXILITES_ADDR_ISR                          0x0c
#define XCONV_AXILITES_ADDR_CHIN_V_DATA                  0x10
#define XCONV_AXILITES_BITS_CHIN_V_DATA                  16
#define XCONV_AXILITES_ADDR_HIN_V_DATA                   0x18
#define XCONV_AXILITES_BITS_HIN_V_DATA                   16
#define XCONV_AXILITES_ADDR_WIN_V_DATA                   0x20
#define XCONV_AXILITES_BITS_WIN_V_DATA                   16
#define XCONV_AXILITES_ADDR_CHOUT_V_DATA                 0x28
#define XCONV_AXILITES_BITS_CHOUT_V_DATA                 16
#define XCONV_AXILITES_ADDR_KX_V_DATA                    0x30
#define XCONV_AXILITES_BITS_KX_V_DATA                    8
#define XCONV_AXILITES_ADDR_KY_V_DATA                    0x38
#define XCONV_AXILITES_BITS_KY_V_DATA                    8
#define XCONV_AXILITES_ADDR_SX_V_DATA                    0x40
#define XCONV_AXILITES_BITS_SX_V_DATA                    8
#define XCONV_AXILITES_ADDR_SY_V_DATA                    0x48
#define XCONV_AXILITES_BITS_SY_V_DATA                    8
#define XCONV_AXILITES_ADDR_MODE_V_DATA                  0x50
#define XCONV_AXILITES_BITS_MODE_V_DATA                  1
#define XCONV_AXILITES_ADDR_RELU_EN_V_DATA               0x58
#define XCONV_AXILITES_BITS_RELU_EN_V_DATA               1
#define XCONV_AXILITES_ADDR_FEATURE_IN_V_DATA            0x60
#define XCONV_AXILITES_BITS_FEATURE_IN_V_DATA            32
#define XCONV_AXILITES_ADDR_FEATURE_IN_PRECISION_V_DATA  0x68
#define XCONV_AXILITES_BITS_FEATURE_IN_PRECISION_V_DATA  4
#define XCONV_AXILITES_ADDR_W_V_DATA                     0x70
#define XCONV_AXILITES_BITS_W_V_DATA                     32
#define XCONV_AXILITES_ADDR_W_PRECISION_V_DATA           0x78
#define XCONV_AXILITES_BITS_W_PRECISION_V_DATA           4
#define XCONV_AXILITES_ADDR_B_V_DATA                     0x80
#define XCONV_AXILITES_BITS_B_V_DATA                     32
#define XCONV_AXILITES_ADDR_B_PRECISION_V_DATA           0x88
#define XCONV_AXILITES_BITS_B_PRECISION_V_DATA           4
#define XCONV_AXILITES_ADDR_FEATURE_OUT_V_DATA           0x90
#define XCONV_AXILITES_BITS_FEATURE_OUT_V_DATA           32
#define XCONV_AXILITES_ADDR_FEATURE_OUT_PRECISION_V_DATA 0x98
#define XCONV_AXILITES_BITS_FEATURE_OUT_PRECISION_V_DATA 4

