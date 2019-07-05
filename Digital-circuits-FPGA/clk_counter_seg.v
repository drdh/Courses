`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:23:55 11/08/2017 
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
module top(clk,rst,duan,wei,data);
	input clk,rst;
	output [3:0]wei,data;
	output [7:0]duan;
	
	cnt count(clk,rst,data);
	coder seg(data,duan,wei);
endmodule

module cnt(clk,rst,data);
	input clk,rst;
	output [3:0]data;
	reg [29:0] count=0;
	assign data=count[29:26];
	always @(posedge clk or posedge rst)
	begin
		if(rst)
			count<=32'h3333_3333;
		else
			count<=count+1;
	end
endmodule

module coder(num,duan,wei);
	input [3:0] num;
	output reg [7:0]duan;
	output  [3:0]wei;
	
	assign	wei=4'b1110;
	
	always @(num)
	begin
		case(num)	
			0:duan=8'b11000000;
			1:duan=8'b11111001;
			2:duan=8'b10100100;
			3:duan=8'b10110000;
			4:duan=8'b10011001;
			5:duan=8'b10010010;
			6:duan=8'b10000010;
			7:duan=8'b11111000;
			8:duan=8'b10000000;
			9:duan=8'b10010000;
			10:duan=8'b10001000;
			11:duan=8'b10000011;
			12:duan=8'b11000110;
			13:duan=8'b10100001;
			14:duan=8'b10000110;
			15:duan=8'b10001110;
		default:duan=8'b11000000;
	endcase
	end
endmodule
