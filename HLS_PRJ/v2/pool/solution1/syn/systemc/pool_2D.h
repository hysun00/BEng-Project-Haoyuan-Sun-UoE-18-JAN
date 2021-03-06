// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2019.1
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

#ifndef _pool_2D_HH_
#define _pool_2D_HH_

#include "systemc.h"
#include "AESL_pkg.h"

#include "pool_mul_64ns_32nbkb.h"
#include "pool_srem_32ns_32cud.h"
#include "pool_2D_buf_V.h"

namespace ap_rtl {

struct pool_2D : public sc_module {
    // Port declarations 31
    sc_in_clk ap_clk;
    sc_in< sc_logic > ap_rst;
    sc_in< sc_logic > ap_start;
    sc_out< sc_logic > ap_done;
    sc_in< sc_logic > ap_continue;
    sc_out< sc_logic > ap_idle;
    sc_out< sc_logic > ap_ready;
    sc_in< sc_lv<32> > ch_div_K_dout;
    sc_in< sc_logic > ch_div_K_empty_n;
    sc_out< sc_logic > ch_div_K_read;
    sc_in< sc_lv<32> > height_in_dout;
    sc_in< sc_logic > height_in_empty_n;
    sc_out< sc_logic > height_in_read;
    sc_in< sc_lv<32> > width_out_dout;
    sc_in< sc_logic > width_out_empty_n;
    sc_out< sc_logic > width_out_read;
    sc_in< sc_lv<32> > Ky_dout;
    sc_in< sc_logic > Ky_empty_n;
    sc_out< sc_logic > Ky_read;
    sc_out< sc_lv<32> > ch_div_K_out_din;
    sc_in< sc_logic > ch_div_K_out_full_n;
    sc_out< sc_logic > ch_div_K_out_write;
    sc_out< sc_lv<32> > width_out_out_din;
    sc_in< sc_logic > width_out_out_full_n;
    sc_out< sc_logic > width_out_out_write;
    sc_out< sc_lv<128> > stream_tp2_V_V_din;
    sc_in< sc_logic > stream_tp2_V_V_full_n;
    sc_out< sc_logic > stream_tp2_V_V_write;
    sc_in< sc_lv<128> > stream_tp_V_V_dout;
    sc_in< sc_logic > stream_tp_V_V_empty_n;
    sc_out< sc_logic > stream_tp_V_V_read;
    sc_signal< sc_logic > ap_var_for_const0;


    // Module declarations
    pool_2D(sc_module_name name);
    SC_HAS_PROCESS(pool_2D);

    ~pool_2D();

    sc_trace_file* mVcdFile;

    pool_2D_buf_V* buf_V_U;
    pool_mul_64ns_32nbkb<1,5,64,32,96>* pool_mul_64ns_32nbkb_U20;
    pool_srem_32ns_32cud<1,36,32,32,32>* pool_srem_32ns_32cud_U21;
    pool_srem_32ns_32cud<1,36,32,32,32>* pool_srem_32ns_32cud_U22;
    sc_signal< sc_logic > ap_done_reg;
    sc_signal< sc_lv<9> > ap_CS_fsm;
    sc_signal< sc_logic > ap_CS_fsm_state1;
    sc_signal< sc_logic > buf_V_ce0;
    sc_signal< sc_lv<128> > buf_V_q0;
    sc_signal< sc_logic > buf_V_ce1;
    sc_signal< sc_logic > buf_V_we1;
    sc_signal< sc_logic > ch_div_K_blk_n;
    sc_signal< sc_logic > height_in_blk_n;
    sc_signal< sc_logic > width_out_blk_n;
    sc_signal< sc_logic > Ky_blk_n;
    sc_signal< sc_logic > ch_div_K_out_blk_n;
    sc_signal< sc_logic > width_out_out_blk_n;
    sc_signal< sc_logic > stream_tp2_V_V_blk_n;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter37;
    sc_signal< bool > ap_block_pp0_stage0;
    sc_signal< sc_lv<1> > icmp_ln144_reg_948;
    sc_signal< sc_logic > stream_tp_V_V_blk_n;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter36;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter35_reg;
    sc_signal< sc_lv<96> > indvar_flatten22_reg_251;
    sc_signal< sc_lv<64> > indvar_flatten_reg_262;
    sc_signal< sc_lv<31> > i_0_i_i_reg_273;
    sc_signal< sc_lv<31> > j_0_i_i_reg_284;
    sc_signal< sc_lv<32> > ch_div_K_read_reg_813;
    sc_signal< bool > ap_block_state1;
    sc_signal< sc_lv<32> > height_in_read_reg_818;
    sc_signal< sc_lv<32> > width_out_read_reg_823;
    sc_signal< sc_lv<32> > Ky_read_reg_830;
    sc_signal< sc_lv<64> > bound_fu_301_p2;
    sc_signal< sc_lv<64> > bound_reg_836;
    sc_signal< sc_logic > ap_CS_fsm_state2;
    sc_signal< sc_logic > ap_CS_fsm_state3;
    sc_signal< sc_lv<96> > grp_fu_313_p2;
    sc_signal< sc_lv<96> > bound4_reg_852;
    sc_signal< sc_logic > ap_CS_fsm_state7;
    sc_signal< sc_lv<1> > icmp_ln125_1_fu_319_p2;
    sc_signal< sc_lv<1> > icmp_ln125_1_reg_857;
    sc_signal< sc_lv<1> > icmp_ln119_fu_339_p2;
    sc_signal< sc_logic > ap_CS_fsm_pp0_stage0;
    sc_signal< bool > ap_block_state8_pp0_stage0_iter0;
    sc_signal< bool > ap_block_state9_pp0_stage0_iter1;
    sc_signal< bool > ap_block_state10_pp0_stage0_iter2;
    sc_signal< bool > ap_block_state11_pp0_stage0_iter3;
    sc_signal< bool > ap_block_state12_pp0_stage0_iter4;
    sc_signal< bool > ap_block_state13_pp0_stage0_iter5;
    sc_signal< bool > ap_block_state14_pp0_stage0_iter6;
    sc_signal< bool > ap_block_state15_pp0_stage0_iter7;
    sc_signal< bool > ap_block_state16_pp0_stage0_iter8;
    sc_signal< bool > ap_block_state17_pp0_stage0_iter9;
    sc_signal< bool > ap_block_state18_pp0_stage0_iter10;
    sc_signal< bool > ap_block_state19_pp0_stage0_iter11;
    sc_signal< bool > ap_block_state20_pp0_stage0_iter12;
    sc_signal< bool > ap_block_state21_pp0_stage0_iter13;
    sc_signal< bool > ap_block_state22_pp0_stage0_iter14;
    sc_signal< bool > ap_block_state23_pp0_stage0_iter15;
    sc_signal< bool > ap_block_state24_pp0_stage0_iter16;
    sc_signal< bool > ap_block_state25_pp0_stage0_iter17;
    sc_signal< bool > ap_block_state26_pp0_stage0_iter18;
    sc_signal< bool > ap_block_state27_pp0_stage0_iter19;
    sc_signal< bool > ap_block_state28_pp0_stage0_iter20;
    sc_signal< bool > ap_block_state29_pp0_stage0_iter21;
    sc_signal< bool > ap_block_state30_pp0_stage0_iter22;
    sc_signal< bool > ap_block_state31_pp0_stage0_iter23;
    sc_signal< bool > ap_block_state32_pp0_stage0_iter24;
    sc_signal< bool > ap_block_state33_pp0_stage0_iter25;
    sc_signal< bool > ap_block_state34_pp0_stage0_iter26;
    sc_signal< bool > ap_block_state35_pp0_stage0_iter27;
    sc_signal< bool > ap_block_state36_pp0_stage0_iter28;
    sc_signal< bool > ap_block_state37_pp0_stage0_iter29;
    sc_signal< bool > ap_block_state38_pp0_stage0_iter30;
    sc_signal< bool > ap_block_state39_pp0_stage0_iter31;
    sc_signal< bool > ap_block_state40_pp0_stage0_iter32;
    sc_signal< bool > ap_block_state41_pp0_stage0_iter33;
    sc_signal< bool > ap_block_state42_pp0_stage0_iter34;
    sc_signal< bool > ap_block_state43_pp0_stage0_iter35;
    sc_signal< bool > ap_block_state44_pp0_stage0_iter36;
    sc_signal< bool > ap_block_state45_pp0_stage0_iter37;
    sc_signal< bool > ap_block_pp0_stage0_11001;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter1_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter2_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter3_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter4_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter5_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter6_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter7_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter8_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter9_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter10_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter11_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter12_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter13_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter14_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter15_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter16_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter17_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter18_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter19_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter20_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter21_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter22_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter23_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter24_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter25_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter26_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter27_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter28_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter29_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter30_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter31_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter32_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter33_reg;
    sc_signal< sc_lv<1> > icmp_ln119_reg_862_pp0_iter34_reg;
    sc_signal< sc_lv<96> > add_ln119_fu_344_p2;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter0;
    sc_signal< sc_lv<31> > select_ln122_5_fu_400_p3;
    sc_signal< sc_lv<31> > select_ln122_5_reg_871;
    sc_signal< sc_lv<31> > select_ln122_6_fu_414_p3;
    sc_signal< sc_lv<31> > select_ln122_6_reg_877;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter1_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter2_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter3_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter4_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter5_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter6_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter7_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter8_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter9_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter10_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter11_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter12_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter13_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter14_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter15_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter16_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter17_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter18_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter19_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter20_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter21_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter22_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter23_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter24_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter25_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter26_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter27_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter28_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter29_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter30_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter31_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter32_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter33_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter34_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter35_reg;
    sc_signal< sc_lv<9> > buf_V_addr_reg_882_pp0_iter36_reg;
    sc_signal< sc_lv<31> > j_fu_427_p2;
    sc_signal< sc_lv<64> > select_ln122_7_fu_439_p3;
    sc_signal< sc_lv<16> > select_ln139_fu_491_p3;
    sc_signal< sc_lv<16> > select_ln139_reg_908;
    sc_signal< sc_lv<16> > select_ln139_1_fu_533_p3;
    sc_signal< sc_lv<16> > select_ln139_1_reg_913;
    sc_signal< sc_lv<16> > select_ln139_2_fu_575_p3;
    sc_signal< sc_lv<16> > select_ln139_2_reg_918;
    sc_signal< sc_lv<16> > select_ln139_3_fu_617_p3;
    sc_signal< sc_lv<16> > select_ln139_3_reg_923;
    sc_signal< sc_lv<16> > select_ln139_4_fu_659_p3;
    sc_signal< sc_lv<16> > select_ln139_4_reg_928;
    sc_signal< sc_lv<16> > select_ln139_5_fu_701_p3;
    sc_signal< sc_lv<16> > select_ln139_5_reg_933;
    sc_signal< sc_lv<16> > select_ln139_6_fu_743_p3;
    sc_signal< sc_lv<16> > select_ln139_6_reg_938;
    sc_signal< sc_lv<16> > select_ln139_7_fu_785_p3;
    sc_signal< sc_lv<16> > select_ln139_7_reg_943;
    sc_signal< sc_lv<1> > icmp_ln144_fu_793_p2;
    sc_signal< bool > ap_block_pp0_stage0_subdone;
    sc_signal< sc_logic > ap_condition_pp0_exit_iter0_state8;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter1;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter2;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter3;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter4;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter5;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter6;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter7;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter8;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter9;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter10;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter11;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter12;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter13;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter14;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter15;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter16;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter17;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter18;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter19;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter20;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter21;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter22;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter23;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter24;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter25;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter26;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter27;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter28;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter29;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter30;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter31;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter32;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter33;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter34;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter35;
    sc_signal< sc_lv<31> > ap_phi_mux_i_0_i_i_phi_fu_277_p4;
    sc_signal< sc_lv<64> > zext_ln135_fu_422_p1;
    sc_signal< sc_lv<128> > tmp_V_fu_799_p9;
    sc_signal< bool > ap_block_pp0_stage0_01001;
    sc_signal< sc_lv<32> > bound_fu_301_p0;
    sc_signal< sc_lv<32> > bound_fu_301_p1;
    sc_signal< sc_lv<64> > grp_fu_313_p0;
    sc_signal< sc_lv<32> > grp_fu_313_p1;
    sc_signal< sc_lv<32> > zext_ln125_fu_330_p1;
    sc_signal< sc_lv<1> > icmp_ln122_fu_350_p2;
    sc_signal< sc_lv<31> > i_fu_324_p2;
    sc_signal< sc_lv<1> > icmp_ln125_fu_334_p2;
    sc_signal< sc_lv<31> > select_ln122_fu_355_p3;
    sc_signal< sc_lv<1> > select_ln122_3_fu_379_p3;
    sc_signal< sc_lv<31> > select_ln122_1_fu_363_p3;
    sc_signal< sc_lv<31> > add_ln144_fu_386_p2;
    sc_signal< sc_lv<31> > select_ln122_2_fu_371_p3;
    sc_signal< sc_lv<31> > add_ln144_1_fu_408_p2;
    sc_signal< sc_lv<31> > select_ln122_4_fu_392_p3;
    sc_signal< sc_lv<64> > add_ln122_fu_433_p2;
    sc_signal< sc_lv<32> > grp_fu_453_p0;
    sc_signal< sc_lv<32> > grp_fu_458_p0;
    sc_signal< sc_lv<32> > grp_fu_453_p2;
    sc_signal< sc_lv<16> > trunc_ln647_1_fu_473_p1;
    sc_signal< sc_lv<16> > trunc_ln647_fu_469_p1;
    sc_signal< sc_lv<1> > icmp_ln895_fu_477_p2;
    sc_signal< sc_lv<1> > icmp_ln139_fu_463_p2;
    sc_signal< sc_lv<16> > select_ln142_fu_483_p3;
    sc_signal< sc_lv<16> > ret_V_1_i_i_fu_509_p4;
    sc_signal< sc_lv<16> > p_Result_i_i_fu_499_p4;
    sc_signal< sc_lv<1> > icmp_ln895_1_fu_519_p2;
    sc_signal< sc_lv<16> > select_ln142_1_fu_525_p3;
    sc_signal< sc_lv<16> > ret_V_2_i_i_fu_551_p4;
    sc_signal< sc_lv<16> > p_Result_8_i_i_fu_541_p4;
    sc_signal< sc_lv<1> > icmp_ln895_2_fu_561_p2;
    sc_signal< sc_lv<16> > select_ln142_2_fu_567_p3;
    sc_signal< sc_lv<16> > ret_V_3_i_i_fu_593_p4;
    sc_signal< sc_lv<16> > p_Result_9_i_i_fu_583_p4;
    sc_signal< sc_lv<1> > icmp_ln895_3_fu_603_p2;
    sc_signal< sc_lv<16> > select_ln142_3_fu_609_p3;
    sc_signal< sc_lv<16> > ret_V_4_i_i_fu_635_p4;
    sc_signal< sc_lv<16> > p_Result_4_i_i_fu_625_p4;
    sc_signal< sc_lv<1> > icmp_ln895_4_fu_645_p2;
    sc_signal< sc_lv<16> > select_ln142_4_fu_651_p3;
    sc_signal< sc_lv<16> > ret_V_5_i_i_fu_677_p4;
    sc_signal< sc_lv<16> > p_Result_5_i_i_fu_667_p4;
    sc_signal< sc_lv<1> > icmp_ln895_5_fu_687_p2;
    sc_signal< sc_lv<16> > select_ln142_5_fu_693_p3;
    sc_signal< sc_lv<16> > ret_V_6_i_i_fu_719_p4;
    sc_signal< sc_lv<16> > p_Result_6_i_i_fu_709_p4;
    sc_signal< sc_lv<1> > icmp_ln895_6_fu_729_p2;
    sc_signal< sc_lv<16> > select_ln142_6_fu_735_p3;
    sc_signal< sc_lv<16> > ret_V_7_i_i_fu_761_p4;
    sc_signal< sc_lv<16> > p_Result_7_i_i_fu_751_p4;
    sc_signal< sc_lv<1> > icmp_ln895_7_fu_771_p2;
    sc_signal< sc_lv<16> > select_ln142_7_fu_777_p3;
    sc_signal< sc_lv<32> > grp_fu_458_p2;
    sc_signal< sc_logic > grp_fu_453_ce;
    sc_signal< sc_logic > grp_fu_458_ce;
    sc_signal< sc_logic > ap_CS_fsm_state46;
    sc_signal< sc_lv<9> > ap_NS_fsm;
    sc_signal< sc_logic > ap_idle_pp0;
    sc_signal< sc_logic > ap_enable_pp0;
    sc_signal< sc_lv<64> > bound_fu_301_p00;
    sc_signal< sc_lv<64> > bound_fu_301_p10;
    sc_signal< sc_lv<96> > grp_fu_313_p00;
    sc_signal< sc_lv<96> > grp_fu_313_p10;
    static const sc_logic ap_const_logic_1;
    static const sc_logic ap_const_logic_0;
    static const sc_lv<9> ap_ST_fsm_state1;
    static const sc_lv<9> ap_ST_fsm_state2;
    static const sc_lv<9> ap_ST_fsm_state3;
    static const sc_lv<9> ap_ST_fsm_state4;
    static const sc_lv<9> ap_ST_fsm_state5;
    static const sc_lv<9> ap_ST_fsm_state6;
    static const sc_lv<9> ap_ST_fsm_state7;
    static const sc_lv<9> ap_ST_fsm_pp0_stage0;
    static const sc_lv<9> ap_ST_fsm_state46;
    static const sc_lv<32> ap_const_lv32_0;
    static const bool ap_const_boolean_1;
    static const bool ap_const_boolean_0;
    static const sc_lv<1> ap_const_lv1_1;
    static const sc_lv<1> ap_const_lv1_0;
    static const sc_lv<32> ap_const_lv32_1;
    static const sc_lv<32> ap_const_lv32_2;
    static const sc_lv<32> ap_const_lv32_6;
    static const sc_lv<32> ap_const_lv32_7;
    static const sc_lv<96> ap_const_lv96_0;
    static const sc_lv<64> ap_const_lv64_0;
    static const sc_lv<31> ap_const_lv31_0;
    static const sc_lv<31> ap_const_lv31_1;
    static const sc_lv<96> ap_const_lv96_1;
    static const sc_lv<31> ap_const_lv31_2;
    static const sc_lv<64> ap_const_lv64_1;
    static const sc_lv<32> ap_const_lv32_10;
    static const sc_lv<32> ap_const_lv32_1F;
    static const sc_lv<32> ap_const_lv32_20;
    static const sc_lv<32> ap_const_lv32_2F;
    static const sc_lv<32> ap_const_lv32_30;
    static const sc_lv<32> ap_const_lv32_3F;
    static const sc_lv<32> ap_const_lv32_40;
    static const sc_lv<32> ap_const_lv32_4F;
    static const sc_lv<32> ap_const_lv32_50;
    static const sc_lv<32> ap_const_lv32_5F;
    static const sc_lv<32> ap_const_lv32_60;
    static const sc_lv<32> ap_const_lv32_6F;
    static const sc_lv<32> ap_const_lv32_70;
    static const sc_lv<32> ap_const_lv32_7F;
    static const sc_lv<32> ap_const_lv32_8;
    // Thread declarations
    void thread_ap_var_for_const0();
    void thread_ap_clk_no_reset_();
    void thread_Ky_blk_n();
    void thread_Ky_read();
    void thread_add_ln119_fu_344_p2();
    void thread_add_ln122_fu_433_p2();
    void thread_add_ln144_1_fu_408_p2();
    void thread_add_ln144_fu_386_p2();
    void thread_ap_CS_fsm_pp0_stage0();
    void thread_ap_CS_fsm_state1();
    void thread_ap_CS_fsm_state2();
    void thread_ap_CS_fsm_state3();
    void thread_ap_CS_fsm_state46();
    void thread_ap_CS_fsm_state7();
    void thread_ap_block_pp0_stage0();
    void thread_ap_block_pp0_stage0_01001();
    void thread_ap_block_pp0_stage0_11001();
    void thread_ap_block_pp0_stage0_subdone();
    void thread_ap_block_state1();
    void thread_ap_block_state10_pp0_stage0_iter2();
    void thread_ap_block_state11_pp0_stage0_iter3();
    void thread_ap_block_state12_pp0_stage0_iter4();
    void thread_ap_block_state13_pp0_stage0_iter5();
    void thread_ap_block_state14_pp0_stage0_iter6();
    void thread_ap_block_state15_pp0_stage0_iter7();
    void thread_ap_block_state16_pp0_stage0_iter8();
    void thread_ap_block_state17_pp0_stage0_iter9();
    void thread_ap_block_state18_pp0_stage0_iter10();
    void thread_ap_block_state19_pp0_stage0_iter11();
    void thread_ap_block_state20_pp0_stage0_iter12();
    void thread_ap_block_state21_pp0_stage0_iter13();
    void thread_ap_block_state22_pp0_stage0_iter14();
    void thread_ap_block_state23_pp0_stage0_iter15();
    void thread_ap_block_state24_pp0_stage0_iter16();
    void thread_ap_block_state25_pp0_stage0_iter17();
    void thread_ap_block_state26_pp0_stage0_iter18();
    void thread_ap_block_state27_pp0_stage0_iter19();
    void thread_ap_block_state28_pp0_stage0_iter20();
    void thread_ap_block_state29_pp0_stage0_iter21();
    void thread_ap_block_state30_pp0_stage0_iter22();
    void thread_ap_block_state31_pp0_stage0_iter23();
    void thread_ap_block_state32_pp0_stage0_iter24();
    void thread_ap_block_state33_pp0_stage0_iter25();
    void thread_ap_block_state34_pp0_stage0_iter26();
    void thread_ap_block_state35_pp0_stage0_iter27();
    void thread_ap_block_state36_pp0_stage0_iter28();
    void thread_ap_block_state37_pp0_stage0_iter29();
    void thread_ap_block_state38_pp0_stage0_iter30();
    void thread_ap_block_state39_pp0_stage0_iter31();
    void thread_ap_block_state40_pp0_stage0_iter32();
    void thread_ap_block_state41_pp0_stage0_iter33();
    void thread_ap_block_state42_pp0_stage0_iter34();
    void thread_ap_block_state43_pp0_stage0_iter35();
    void thread_ap_block_state44_pp0_stage0_iter36();
    void thread_ap_block_state45_pp0_stage0_iter37();
    void thread_ap_block_state8_pp0_stage0_iter0();
    void thread_ap_block_state9_pp0_stage0_iter1();
    void thread_ap_condition_pp0_exit_iter0_state8();
    void thread_ap_done();
    void thread_ap_enable_pp0();
    void thread_ap_idle();
    void thread_ap_idle_pp0();
    void thread_ap_phi_mux_i_0_i_i_phi_fu_277_p4();
    void thread_ap_ready();
    void thread_bound_fu_301_p0();
    void thread_bound_fu_301_p00();
    void thread_bound_fu_301_p1();
    void thread_bound_fu_301_p10();
    void thread_bound_fu_301_p2();
    void thread_buf_V_ce0();
    void thread_buf_V_ce1();
    void thread_buf_V_we1();
    void thread_ch_div_K_blk_n();
    void thread_ch_div_K_out_blk_n();
    void thread_ch_div_K_out_din();
    void thread_ch_div_K_out_write();
    void thread_ch_div_K_read();
    void thread_grp_fu_313_p0();
    void thread_grp_fu_313_p00();
    void thread_grp_fu_313_p1();
    void thread_grp_fu_313_p10();
    void thread_grp_fu_453_ce();
    void thread_grp_fu_453_p0();
    void thread_grp_fu_458_ce();
    void thread_grp_fu_458_p0();
    void thread_height_in_blk_n();
    void thread_height_in_read();
    void thread_i_fu_324_p2();
    void thread_icmp_ln119_fu_339_p2();
    void thread_icmp_ln122_fu_350_p2();
    void thread_icmp_ln125_1_fu_319_p2();
    void thread_icmp_ln125_fu_334_p2();
    void thread_icmp_ln139_fu_463_p2();
    void thread_icmp_ln144_fu_793_p2();
    void thread_icmp_ln895_1_fu_519_p2();
    void thread_icmp_ln895_2_fu_561_p2();
    void thread_icmp_ln895_3_fu_603_p2();
    void thread_icmp_ln895_4_fu_645_p2();
    void thread_icmp_ln895_5_fu_687_p2();
    void thread_icmp_ln895_6_fu_729_p2();
    void thread_icmp_ln895_7_fu_771_p2();
    void thread_icmp_ln895_fu_477_p2();
    void thread_j_fu_427_p2();
    void thread_p_Result_4_i_i_fu_625_p4();
    void thread_p_Result_5_i_i_fu_667_p4();
    void thread_p_Result_6_i_i_fu_709_p4();
    void thread_p_Result_7_i_i_fu_751_p4();
    void thread_p_Result_8_i_i_fu_541_p4();
    void thread_p_Result_9_i_i_fu_583_p4();
    void thread_p_Result_i_i_fu_499_p4();
    void thread_ret_V_1_i_i_fu_509_p4();
    void thread_ret_V_2_i_i_fu_551_p4();
    void thread_ret_V_3_i_i_fu_593_p4();
    void thread_ret_V_4_i_i_fu_635_p4();
    void thread_ret_V_5_i_i_fu_677_p4();
    void thread_ret_V_6_i_i_fu_719_p4();
    void thread_ret_V_7_i_i_fu_761_p4();
    void thread_select_ln122_1_fu_363_p3();
    void thread_select_ln122_2_fu_371_p3();
    void thread_select_ln122_3_fu_379_p3();
    void thread_select_ln122_4_fu_392_p3();
    void thread_select_ln122_5_fu_400_p3();
    void thread_select_ln122_6_fu_414_p3();
    void thread_select_ln122_7_fu_439_p3();
    void thread_select_ln122_fu_355_p3();
    void thread_select_ln139_1_fu_533_p3();
    void thread_select_ln139_2_fu_575_p3();
    void thread_select_ln139_3_fu_617_p3();
    void thread_select_ln139_4_fu_659_p3();
    void thread_select_ln139_5_fu_701_p3();
    void thread_select_ln139_6_fu_743_p3();
    void thread_select_ln139_7_fu_785_p3();
    void thread_select_ln139_fu_491_p3();
    void thread_select_ln142_1_fu_525_p3();
    void thread_select_ln142_2_fu_567_p3();
    void thread_select_ln142_3_fu_609_p3();
    void thread_select_ln142_4_fu_651_p3();
    void thread_select_ln142_5_fu_693_p3();
    void thread_select_ln142_6_fu_735_p3();
    void thread_select_ln142_7_fu_777_p3();
    void thread_select_ln142_fu_483_p3();
    void thread_stream_tp2_V_V_blk_n();
    void thread_stream_tp2_V_V_din();
    void thread_stream_tp2_V_V_write();
    void thread_stream_tp_V_V_blk_n();
    void thread_stream_tp_V_V_read();
    void thread_tmp_V_fu_799_p9();
    void thread_trunc_ln647_1_fu_473_p1();
    void thread_trunc_ln647_fu_469_p1();
    void thread_width_out_blk_n();
    void thread_width_out_out_blk_n();
    void thread_width_out_out_din();
    void thread_width_out_out_write();
    void thread_width_out_read();
    void thread_zext_ln125_fu_330_p1();
    void thread_zext_ln135_fu_422_p1();
    void thread_ap_NS_fsm();
};

}

using namespace ap_rtl;

#endif
