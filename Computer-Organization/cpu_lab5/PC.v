`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:20:52 04/15/2018 
// Design Name: 
// Module Name:    PC 
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
module PC(
	input clk,
	input [31:0]pc_i,
	output [31:0]pc_o
    );
	reg [31:0]PC;
	assign pc_o=PC;
	always@(posedge clk)
		PC=pc_i;
		
	initial
		PC=0;
endmodule
