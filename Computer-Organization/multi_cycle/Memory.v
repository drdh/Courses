`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:04:32 04/16/2018 
// Design Name: 
// Module Name:    Memory 
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
module Memory(
	input clk,
	input WE,
	input [5:0]A,
	input [31:0]WD,
	output [31:0]RD
    );
	reg [31:0]Mem[0:63];
	assign RD=Mem[(A>>2)];
	always@(posedge clk)
		if(WE)
			Mem[(A>>2)]=WD;
	
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
		Mem[0]=32'h00242820;
		Mem[1]=32'hac250004;
		Mem[2]=32'h8c260004;
		Mem[3]=32'h10a60001;
		Mem[4]=32'h00210820;
		Mem[5]=32'h00000000;
	end

endmodule
