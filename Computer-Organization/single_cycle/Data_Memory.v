`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:11:18 04/15/2018 
// Design Name: 
// Module Name:    Data_Memory 
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
module Data_Memory(
	input clk,
	input MemWrite,
	input [5:0]A,
	input [31:0]WD,
	output [31:0]RD
    );
	reg [31:0]Mem[0:63];
	assign RD=Mem[(A>>2)];
	always@(posedge clk)
		if(MemWrite)
			Mem[(A>>2)]=WD;
endmodule
