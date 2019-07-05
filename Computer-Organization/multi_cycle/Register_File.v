`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    16:58:43 04/15/2018 
// Design Name: 
// Module Name:    Register_File 
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
module Register_File(
	input clk,
	input [4:0]A1,
	input [4:0]A2,
	input [4:0]A3,
	input [31:0]WD3,
	input RegWrite,
	output [31:0]RD1,
	output [31:0]RD2
    );
	reg [31:0]reg_file[0:31];
	assign RD1=RegWrite ? (A1==A3 ? WD3 : reg_file[A1]):reg_file[A1];
	assign RD2=RegWrite ? (A2==A3 ? WD3 : reg_file[A2]):reg_file[A2];
	
	always@(posedge clk)
	if(RegWrite)
		reg_file[A3]=WD3;
		
	initial
	begin 
		reg_file[0]=0;
		reg_file[1]=32;
		reg_file[2]=2;
		reg_file[3]=3;
		reg_file[4]=4;		
		$monitor($time,"$0=%d",reg_file[0]);
	end 
endmodule
