`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    19:01:32 03/28/2018 
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
	reg signed [31:0]a,b;
	wire signed [31:0]out;
	reg [4:0]op;
	
	ALU aluTest(a,b,op,out);
	
	initial
	$monitor($time,"a=%b,b=%b,op=%h,out=%b",a,b,op,out);
	
	initial 
	begin
	#10 a=1; b=1;op=0;
	#10 a=11;b=12;op=1;
	#10 a=12;b=10;op=2;
	#10 a=10;b=12;op=2;
	#10 a=1;b=3;op=3;
	#10 a=1;b=2;op=3;
	#10 a=1;b=3;op=4;
	#10 a=1;b=2;op=4;
	#10 a=1;b=3;op=5;
	#10 a=1;b=2;op=5;
	#10 a=1;b=3;op=6;
	#10 a=1;b=2;op=6;
	#100 $finish;
	end


endmodule
