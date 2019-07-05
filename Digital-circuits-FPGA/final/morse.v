`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    12:45:30 11/29/2017 
// Design Name: 
// Module Name:    morse 
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
module to_morse(data,code,width);
	input [5:0]data;
	output reg[4:0]code;
	output reg[2:0]width; 
	always@(data)
	case(data)
	0:begin code=5'b11111;	width=3'd5;	end
	1:begin code=5'b01111;	width=3'd5;	end
	2:begin code=5'b00111;	width=3'd5;	end
	3:begin code=5'b00011;	width=3'd5;	end
	4:begin code=5'b00001;	width=3'd5;	end
	5:begin code=5'b00000;	width=3'd5;	end
	6:begin code=5'b10000;	width=3'd5;	end
	7:begin code=5'b11000;	width=3'd5;	end
	8:begin code=5'b11100;	width=3'd5;	end
	9:begin code=5'b11110;	width=3'd5;	end
  10:begin code=5'b01;		width=3'd2;	end//A
  11:begin code=5'b1000;	width=3'd4;	end//B
  12:begin code=5'b1010;	width=3'd4;	end//C
  13:begin code=5'b100;		width=3'd3;	end//d
  14:begin code=5'b0;		width=3'd1;	end//E
  15:begin code=5'b0010;	width=3'd4;	end//F
  16:begin code=5'b110;		width=3'd3;	end//G
  17:begin code=5'b0000;	width=3'd4;	end//H
  18:begin code=5'b00; 		width=3'd2;	end//I
  19:begin code=5'b0111;	width=3'd4;	end//J
  20:begin code=5'b101;		width=3'd3;	end//K
  21:begin code=5'b0100;	width=3'd4;	end//L
  22:begin code=5'b11;		width=3'd2;	end//M
  23:begin code=5'b10;		width=3'd2;	end//n
  24:begin code=5'b111;		width=3'd3;	end//o
  25:begin code=5'b0110;	width=3'd4;	end//P
  26:begin code=5'b1101;	width=3'd4;	end//q
  27:begin code=5'b010;		width=3'd3;	end//r
  28:begin code=5'b000;		width=3'd3;	end//S
  29:begin code=5'b1;		width=3'd1;	end//t
  30:begin code=5'b001;		width=3'd3;	end//U
  31:begin code=5'b0001;	width=3'd4;	end//v
  32:begin code=5'b011;		width=3'd3;	end//w
  33:begin code=5'b1001;	width=3'd4;	end//x
  34:begin code=5'b1011;	width=3'd4;	end//y
  35:begin code=5'b1100;	width=3'd4;	end//z
	default:begin 
	  code=5'b11111;			width=3'd5;	end//0
	endcase
endmodule
