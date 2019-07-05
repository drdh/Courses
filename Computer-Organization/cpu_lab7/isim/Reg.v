`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:35:26 05/12/2018 
// Design Name: 
// Module Name:    Reg 
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
module Reg
#(parameter width=32)
(	input clk,
	input reset,
	input enable,
	input [width-1:0]in,
	output reg[width-1:0]out 
    );
	always@(posedge clk)
	if(reset)
		out<={width{1'b0}};
	else if(enable)
		out<=in;
	else
		out<=out;
	initial
		out=0;
	
endmodule

/*
Reg	#(.width(32)) (.clk(clk),.reset(0),.enable(1),.in(),.out());
*/
