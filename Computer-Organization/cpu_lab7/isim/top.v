`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    21:52:46 05/27/2018 
// Design Name: 
// Module Name:    top 
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
module top(
	input clk,
	input disp_en,
	input [4:0]disp_addr,
	output [7:0]seg,
	output [3:0]an);
	
wire [15:0]data;


reg [26:0]count=0;
always@(posedge clk)
	count<=count+1;
CPU cpu_unit(count[25],data,disp_en,disp_addr);


//CPU cpu_unit(clk,data,disp_en,disp_addr);
display display_unit(clk,data,seg,an);

endmodule
