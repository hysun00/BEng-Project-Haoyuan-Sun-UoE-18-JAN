#ifndef __CONV_H__
#define __CONV_H__

#include <ap_int.h>
#include <iostream>

using namespace std;

typedef float Dtype_f;
typedef float Dtype_w;
typedef float Dtype_mul;
typedef float Dtype_acc;

void Conv
(
	ap_uint<16> CHin,
	ap_uint<16> Hin,
	ap_uint<16> Win,
	ap_uint<16> CHout,
	ap_uint<8> Kx,
	ap_uint<8> Ky,
	ap_uint<8> Sx,
	ap_uint<8> Sy,
	ap_uint<1> mode,//mode: 0:VALID, 1:SAME
	ap_uint<1> relu_en,
	Dtype_f feature_in[],
	Dtype_w W[],
	Dtype_w bias[],
	Dtype_f feature_out[]
);

#endif
