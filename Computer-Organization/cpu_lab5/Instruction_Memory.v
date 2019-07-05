`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:08:32 04/15/2018 
// Design Name: 
// Module Name:    Instruction_Memory 
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
module Instruction_Memory(
	input [31:0]PA,
	output [31:0]RD
    );
	//reg [31:0]Instr[0:63];
	wire [5:0]A;
	assign A=PA>>2;
	//assign RD=Instr[A];
	
	InstrRam Instr(.a(A),.spo(RD));
	
	/*
	initial
	begin
	/*
	add $5,$1,$4 
	sw $5,4($1)
	lw $6,4($1)
	beq $5,$6,label
	add $1,$1,$1
	label: nop
	*/
	/*	Instr[0]=32'h00242820;
		Instr[1]=32'hac250004;
		Instr[2]=32'h8c260004;
		Instr[3]=32'h10a60001;
		Instr[4]=32'h00210820;
		Instr[5]=32'h00000000;
	*/
	/*
	Instr[0]=32'h20080000;
	Instr[1]=32'h200d0050;
	Instr[2]=32'h8dad0000;
	Instr[3]=32'h200b0054;
	Instr[4]=32'h8d6b0000;
	Instr[5]=32'h200c0054;
	Instr[6]=32'h8d8c0004;
	
	Instr[7]=32'had0b0000;
	Instr[8]=32'had0c0004;
	Instr[9]=32'h21a9fffe;
	
	Instr[10]=32'h8d0b0000;
	Instr[11]=32'h8d0c0004;
	Instr[12]=32'h016c5020;
	Instr[13]=32'had0a0008;
	Instr[14]=32'h21080004;
	Instr[15]=32'h2129ffff;
	Instr[16]=32'h1d20fff9;
	Instr[17]=32'h08000011;
	*/
	/*
	end
	*/
endmodule
