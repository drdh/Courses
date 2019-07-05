`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    19:31:10 03/28/2018 
// Design Name: 
// Module Name:    ftop 
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
module ftop;
	reg [31:0]a,b;
	wire [31:0]out;
	
	wire [31:0]out1,out2,out3;
	ALU a1(a,b,5'h01,out1);
	ALU a2(b,out1,5'h01,out2);
	ALU a3(out1,out2,5'h01,out3);
	ALU a4(out2,out3,5'h01,out);
	
	initial
	$monitor($time,"a=%d,b=%d,out=%d",a,b,out);
	
	initial
	begin
	#10 a=2;b=2;
	#10 $finish;
	end


endmodule
