`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    16:52:15 04/15/2018 
// Design Name: 
// Module Name:    ALU 
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
module ALU(
	input [2:0]ALUCtl,
	input [31:0]A,
	input [31:0]B,
	output reg [31:0]ALUOut,
	output Zero
    );
	assign Zero=(ALUOut==0);
	always@(*)
		case(ALUCtl)
		0:ALUOut<=A&B;
		1:ALUOut<=A|B;
		2:ALUOut<=A+B;
		6:ALUOut<=A-B;
		7:ALUOut<=A<B?1:0;
		12:ALUOut<=~(A|B);
		default:ALUOut<=0;
		endcase
endmodule
