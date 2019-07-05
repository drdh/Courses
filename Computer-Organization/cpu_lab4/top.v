`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    12:23:11 04/18/2018 
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
module top;
	reg clk;
	reg rst;
	wire [31:0]out;
	
	Control ctl_unit(clk,rst,out);
	
	initial
	begin
		clk=1;
		rst=0;
		#10 rst=1;
		#10 rst=0;
		#10;
		forever #2 clk=~clk;
	end 
	
	always@(out)
		if(out==32'hffff_ffff)
			$finish;


endmodule
