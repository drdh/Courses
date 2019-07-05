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
	input [5:0]A,
	output [31:0]RD
    );
	reg [31:0]Instr[0:63];
	assign RD=Instr[(A>>2)];
	
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
		Instr[0]=32'h00242820;
		Instr[1]=32'hac250004;
		Instr[2]=32'h8c260004;
		Instr[3]=32'h10a60001;
		Instr[4]=32'h00210820;
		Instr[5]=32'h00000000;
	end
endmodule
