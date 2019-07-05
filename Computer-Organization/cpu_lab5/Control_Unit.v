`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    16:06:14 04/15/2018 
// Design Name: 
// Module Name:    Control_Unit 
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
module Control_Unit(
	input [5:0]Op,
	input [5:0]Funct,
	output reg RegWrite,
	output reg RegDst,
	output reg ALUSrc,
	output reg [2:0]ALUControl,
	output reg Branch,
	output reg MemWrite,
	output reg MemtoReg,
	output reg Jump,
	output reg bgtz
    );
	always@(*)
	begin
		RegDst=0;
		ALUSrc=0;
		MemtoReg=0;
		RegWrite=0;
		MemWrite=0;
		Branch=0;
		ALUControl=0;
		Jump=0;
		bgtz=0;
	casex({Op,Funct})
		12'b000000_100000: begin//add
		RegDst=1;
		RegWrite=1;
		ALUControl=3'b010;
		end
		12'b000000_100010: begin//sub
		RegDst=1;
		RegWrite=1;
		ALUControl=3'b110;
		end
		12'b000000_100100: begin//and
		RegDst=1;
		RegWrite=1;
		ALUControl=3'b000;
		end
		12'b000000_100101: begin//or
		RegDst=1;
		RegWrite=1;
		ALUControl=3'b001;
		end
		12'b000000_101010: begin//slt
		RegDst=1;
		RegWrite=1;
		ALUControl=3'b111;
		end
		12'b100011_xxxxxx: begin//lw
		ALUSrc=1;
		MemtoReg=1;
		RegWrite=1;
		ALUControl=3'b010;
		end
		12'b101011_xxxxxx: begin//sw
		ALUSrc=1;
		MemWrite=1;
		ALUControl=3'b010;
		end
		12'b000100_xxxxxx: begin//beq
		Branch=1;
		ALUControl=3'b110;
		end
		12'b001000_xxxxxx: begin//addi (lw)
		ALUSrc=1;
		//MemtoReg=1;
		RegWrite=1;
		ALUControl=3'b010;
		end
		12'b000111_xxxxxx: begin//bgtz (beq)
		//Branch=1;
		bgtz=1;
		ALUControl=3'b110;
		end
		12'b000010_xxxxxx: begin//j
		Jump=1;
		end
		default:;
	endcase
	end
endmodule
