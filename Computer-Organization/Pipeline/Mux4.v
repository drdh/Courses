`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:18:56 05/12/2018 
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
#(parameter width=32)
(	input [1:0]sel,
	input [width-1:0]in0,
	input [width-1:0]in1,
	input [width-1:0]in2,
	input [width-1:0]in3,
	output reg[width-1:0]out
    );
always@(*)
begin
	case(sel)
		2'b00:out<=in0;
		2'b01:out<=in1;
		2'b10:out<=in2;
		2'b11:out<=in3;
		default:out<=0;
	endcase
end 

endmodule

/*
Mux4 #(.width(32)) (.sel(),.in0(),.in1(),.in2(),.in3(),.out());
*/