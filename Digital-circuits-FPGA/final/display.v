`timescale 1ns / 1ps

module display(clk,data1,data2,seg,an);
	input clk;
	input [5:0]data1,data2;		//0-9 a-z
	output reg [3:0]an;
	output reg [7:0]seg;
	
	reg [5:0]disp;
	
	reg [18:0]count=0;
	always@(posedge clk)			
		count<=count+1;	
		
	always@(posedge clk)
		case(count[18])
		1'b0:begin
			an=4'b0111;
			disp=data1;
		end
		1'b1:begin
			an=4'b1110;
			disp=data2;
		end
	endcase
	
	always @(disp)
	case(disp)
		0:seg=8'b11000000;	//0
		1:seg=8'b11111001;	//1
		2:seg=8'b10100100;	//2
		3:seg=8'b10110000;	//3
		4:seg=8'b10011001;	//4
		5:seg=8'b10010010;	//5
		6:seg=8'b10000010;	//6
		7:seg=8'b11111000;	//7
		8:seg=8'b10000000;	//8
		9:seg=8'b10010000;	//9
	   10:seg=8'b10001000;	//A
	   11:seg=8'b10000011;	//b
	   12:seg=8'b11000110;	//C
	   13:seg=8'b10100001;	//d
	   14:seg=8'b10000110;	//E
	   15:seg=8'b10001110;	//F
	   16:seg=8'b11000010;	//G
	   17:seg=8'b10001001;	//H
	   18:seg=8'b11101111;	//i 
	   19:seg=8'b11110001;	//J
	   20:seg=8'b10000101;	//K
	   21:seg=8'b11000111;	//L
	   22:seg=8'b10101010;	//M
	   23:seg=8'b10101011;	//N
	   24:seg=8'b10100011;	//o
	   25:seg=8'b10001100;	//P
	   26:seg=8'b10011000;	//q
	   27:seg=8'b10101111;	//r
	   28:seg=8'b10011011;	//S
	   29:seg=8'b10000111;	//t 
	   30:seg=8'b11000001;	//U
	   31:seg=8'b10011101;	//v
	   32:seg=8'b10010101;	//w
	   33:seg=8'b11001001;	//x
	   34:seg=8'b10010001;	//y
	   35:seg=8'b10110110;	//z
  default:seg=8'b11000000;
	endcase
endmodule
