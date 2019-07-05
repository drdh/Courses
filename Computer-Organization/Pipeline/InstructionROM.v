`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    13:31:28 05/12/2018 
// Design Name: 
// Module Name:    InstructionROM 
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
module InstructionROM(
	input [31:0]addr,
	output [31:0]dout
    );
	reg [31:0]Instr[0:63];
	
	wire [5:0]A;
	assign A=(addr>>2);
	assign dout=Instr[A];
	
	initial
	begin
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
	end 
	
endmodule
