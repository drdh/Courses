`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:25:47 11/29/2017 
// Design Name: 
// Module Name:    ledplay 
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
module ledplay(clk,code,width,led);
input clk;
input [4:0]code;
input [2:0]width;
output reg[7:0]led;
reg [25:0]count=0;


always@(posedge clk)
	count<=count+1'b1;
		
always@(code or count[25])
begin
	case(width)
	1:begin	
		led[7]=code[0]|count[25];	
		led[6:0]=0;
	end
	2:begin
		led[7:6]=code[1:0]|{2{count[25]}};
		led[5:0]=0;
	end
	3:begin
		led[7:5]=code[2:0]|{3{count[25]}};
		led[4:0]=0;
	end
	4:begin
		led[7:4]=code[3:0]|{4{count[25]}};
		led[3:0]=0;
	end
	5:begin
		led[7:3]=code[4:0]|{5{count[25]}};
		led[2:0]=0;
	end
	default:begin
		led[7:0]=0;
	end
	endcase
end

endmodule
