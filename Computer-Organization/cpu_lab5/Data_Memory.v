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
	input [31:0]PA,
	input [31:0]WD,
	output [31:0]RD
    );
	//reg [31:0]Mem[0:63];
	wire [5:0]A;
	assign A=PA>>2;
	//assign RD=Mem[A];
	
	
	MemRam Mem(.a(A),.d(WD),.dpra(A),.we(MemWrite),.clk(clk),.dpo(RD));
	/*
	always@(posedge clk)
		if(MemWrite)
			Mem[A]=WD;
	*/
	/*
	initial
	begin
	Mem[0]=32'h00000000;
	/*Mem[1]=32'h00000001;
	Mem[2]=32'h00000002;
	Mem[3]=32'h00000000;
	Mem[4]=32'h00000004;
	Mem[5]=32'h00000005;
	Mem[6]=32'h00000006;
	Mem[7]=32'h00000007;
	Mem[8]=32'h00000008;
	Mem[9]=32'h00000009;
	Mem[10]=32'h0000000A;
	Mem[11]=32'h0000000B;
	Mem[12]=32'h0000000C;
	Mem[13]=32'h0000000D;
	Mem[14]=32'h0000000E;
	Mem[15]=32'h0000000F;
	*/
	/*
	Mem[20]=32'h00000014;
	Mem[21]=32'h00000003;
	Mem[22]=32'h00000003;
	
	end
	*/
	/*initial
	begin
	Mem[0]=32'h20080000;
	Mem[1]=32'h200d0050;
	Mem[2]=32'h8dad0000;
	Mem[3]=32'h200b0054;
	Mem[4]=32'h8d6b0000;
	Mem[5]=32'h200c0054;
	Mem[6]=32'h8d8c0004;
	Mem[7]=32'had0b0000;
	Mem[8]=32'had0c0004;
	Mem[9]=32'h21a9fffe;
	Mem[10]=32'h8d0b0000;
	Mem[11]=32'h8d0c0004;
	Mem[12]=32'h016c5020;
	Mem[13]=32'had0a0008;
	Mem[14]=32'h21080004;
	Mem[15]=32'h2129ffff;
	Mem[16]=32'h1d20fff9;
	Mem[17]=32'h08000011;
	end
	*/
endmodule
