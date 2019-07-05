`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    21:28:19 05/28/2018 
// Design Name: 
// Module Name:    Sequence 
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
module Sequence(
	input clk,
	input en,
	output [4:0]out
    );
	reg [4:0]addr=0;
	assign out=addr;
	
	reg [25:0]count=0;
	always@(posedge clk)
	if(en)
		count<=count+1;
	
	always@(posedge count[25])
		addr<=addr+1;

endmodule
