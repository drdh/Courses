`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    15:17:06 04/16/2018 
// Design Name: 
// Module Name:    Mux4 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module Mux4
	#(parameter DATA_SIZE=32)
	(
	input [1:0]sel,
	input [DATA_SIZE-1:0]S0,
	input [DATA_SIZE-1:0]S1,
	input [DATA_SIZE-1:0]S2,
	input [DATA_SIZE-1:0]S3,
	output reg[DATA_SIZE-1:0]out
    );
	always@(*)
	case(sel)
	2'b00:out=S0;
	2'b01:out=S1;
	2'b10:out=S2;
	2'b11:out=S3;
	default:out=0;
	endcase
endmodule
