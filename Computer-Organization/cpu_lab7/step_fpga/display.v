`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:33:22 05/27/2018 
// Design Name: 
// Module Name:    display 
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
module display(clk,data,seg,an);
	input clk;
	input [15:0]data;
	output reg [3:0]an;
	output reg [7:0]seg;
	
	reg [3:0]disp;
	
	reg [18:0]count=0;
	always@(posedge clk)			
		count<=count+1;	
		
	always@(posedge clk)
		case(count[18:17])
		2'b00:begin
			an=4'b1110;
			disp=data[3:0];
		end
		2'b01:begin
			an=4'b1101;
			disp=data[7:4];
		end
		2'b10:begin
			an=4'b1011;
			disp=data[11:8];
		end
		2'b11:begin
			an=4'b0111;
			disp=data[15:12];
		end
	endcase
	
	always @(disp)
	case(disp)
		0:seg=8'b11000000;
		1:seg=8'b11111001;
		2:seg=8'b10100100;
		3:seg=8'b10110000;
		4:seg=8'b10011001;
		5:seg=8'b10010010;
		6:seg=8'b10000010;
		7:seg=8'b11111000;
		8:seg=8'b10000000;
		9:seg=8'b10010000;
	  10:seg=8'b10001000;
	  11:seg=8'b10000011;
	  12:seg=8'b11000110;
	  13:seg=8'b10100001;
	  14:seg=8'b10000110;
	  15:seg=8'b10001110;
default:seg=8'b11000000;
	endcase
endmodule
