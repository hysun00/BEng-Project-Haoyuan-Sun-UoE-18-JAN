// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __Pool_uitofp_32ns_dEe__HH__
#define __Pool_uitofp_32ns_dEe__HH__
#include "ACMP_uitofp.h"
#include <systemc>

template<
    int ID,
    int NUM_STAGE,
    int din0_WIDTH,
    int dout_WIDTH>
SC_MODULE(Pool_uitofp_32ns_dEe) {
    sc_core::sc_in_clk clk;
    sc_core::sc_in<sc_dt::sc_logic> reset;
    sc_core::sc_in<sc_dt::sc_logic> ce;
    sc_core::sc_in< sc_dt::sc_lv<din0_WIDTH> >   din0;
    sc_core::sc_out< sc_dt::sc_lv<dout_WIDTH> >   dout;



    ACMP_uitofp<ID, 6, din0_WIDTH, dout_WIDTH> ACMP_uitofp_U;

    SC_CTOR(Pool_uitofp_32ns_dEe):  ACMP_uitofp_U ("ACMP_uitofp_U") {
        ACMP_uitofp_U.clk(clk);
        ACMP_uitofp_U.reset(reset);
        ACMP_uitofp_U.ce(ce);
        ACMP_uitofp_U.din0(din0);
        ACMP_uitofp_U.dout(dout);

    }

};

#endif //
