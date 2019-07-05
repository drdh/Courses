`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:11:13 04/16/2018 
// Design Name: 
// Module Name:    Register 
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
module Register
	#(
	parameter DATA_SIZE=32
	)
	(
	input clk,
	input EN,
	input [DATA_SIZE-1:0]din,
	output [DATA_SIZE-1:0]dout
    );
	reg [DATA_SIZE-1:0]data;
	assign dout=data;
	always@(posedge clk)
		if(EN)
			data=din;
			
	initial
		data=0;
endmodule
