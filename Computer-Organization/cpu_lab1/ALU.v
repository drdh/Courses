`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:47:15 03/28/2018 
// Design Name: 
// Module Name:    ALU 
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
module ALU(
	input signed [31:0]alu_a,
	input signed [31:0]alu_b,
	input [4:0]alu_op,
	output signed [31:0] alu_out
    );
	 parameter A_NOP=5'h00;
	 parameter A_ADD=5'h01;
	 parameter A_SUB=5'h02;
	 parameter A_AND=5'h03;
	 parameter A_OR= 5'h04;
	 parameter A_XOR=5'h05;
	 parameter A_NOR=5'h06;
	 
	 reg signed [31:0] out;
	 assign alu_out=out;
	 always@(*)
	 case(alu_op)
	 A_NOP:	out<=32'b0;
	 A_ADD:	out<=alu_a+alu_b;
	 A_SUB:  out<=alu_a-alu_b;
	 A_AND:	out<=alu_a&alu_b;
	 A_OR :  out<=alu_a|alu_b;
	 A_XOR:  out<=alu_a^alu_b;
	 A_NOR:	out<=alu_a^~alu_b;
	 default:out<=32'b0;
	 endcase
endmodule
