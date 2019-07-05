`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:57:36 05/12/2018 
// Design Name: 
// Module Name:    DataRam 
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
module DataRam(
	input clk,
	input [31:0]addr,
	input [31:0]din,
	input we,
	output [31:0]dout
    );
	reg [31:0]Mem[0:63];
	wire [5:0]A;
	assign A=(addr>>2);
	assign dout=Mem[A];
	
	always@(posedge clk)
	begin
		//dout<=Mem[A];
		if(we)
			Mem[A]<=din;
	end 
	
	initial
	begin
	Mem[0]=32'h00000000;
	/*
	Mem[1]=32'h00000001;
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
	Mem[20]=32'h00000014;
	Mem[21]=32'h00000003;
	Mem[22]=32'h00000003;
	
	end
	
endmodule
