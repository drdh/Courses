`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    23:37:06 04/15/2018 
// Design Name: 
// Module Name:    Control 
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
module Control(
	input  clk,
	input [5:0]Op,
	input [5:0]Funct,
	output reg RegWrite,
	output reg ALUSrcA,
	output reg[1:0] ALUSrcB,
	output reg[2:0]ALUControl,
	output reg PCSrc,
	output reg Branch,
	output reg PCWrite,
	output reg IorD,
	output reg MemWrite,
	output reg IRWrite,
	output reg RegDst,
	output reg MemtoReg
    );
	localparam S0=4'h0;
	localparam S1=4'h1;
	localparam S2=4'h2;
	localparam S3=4'h3;
	localparam S4=4'h4;
	localparam S5=4'h5;
	localparam S6=4'h6;
	localparam S7=4'h7;
	localparam S8=4'h8;
	
	localparam R=6'b000000;
	localparam LW=6'b100011;
	localparam SW=6'b101011;
	localparam BEQ=6'b000100;
	
	reg [3:0]state=0,next_state=0;
	//reg [1:0]ALUOp;
	
	always@(posedge clk)
		state<=next_state;
		
	always@(state)
	begin
		RegWrite=0;
		ALUSrcA=0;
		ALUSrcB=0;
		ALUControl=0;
		PCSrc=0;
		Branch=0;
		PCWrite=0;
		IorD=0;
		MemWrite=0;
		IRWrite=0;
		RegDst=0;
		MemtoReg=0;
		case(state)
			S0:begin//Fetch
			IorD=0;
			ALUSrcA=0;
			ALUSrcB=2'b01;
			ALUControl=3'b010;
			PCSrc=0;
			IRWrite=1;
			PCWrite=1;
			end
			S1:begin//decode
			ALUSrcA=0;
			ALUSrcB=2'b11;
			ALUControl=3'b010;
			end
			S2:begin//MemAdr
			ALUSrcA=1;
			ALUSrcB=2'b10;
			ALUControl=3'b010;
			end
			S3:begin//MemRead
			IorD=1;
			end
			S4:begin//MemWriteBack
			RegDst=0;
			MemtoReg=1;
			RegWrite=1;
			end
			S5:begin//MemWrite
			IorD=1;
			MemWrite=1;
			end
			S6:begin//Excute
			ALUSrcA=1;
			ALUSrcB=2'b00;
			case(Funct)
			6'b100000:ALUControl=3'b010;//Add 
			6'b100010:ALUControl=3'b110;//Sub
			6'b100100:ALUControl=3'b000;//And
			6'b100101:ALUControl=3'b001;//or
			6'b101010:ALUControl=3'b111;//slt
			default:ALUControl=0;
			endcase
			end
			S7:begin//ALUWriteBack
			RegDst=1;
			MemtoReg=0;
			RegWrite=1;
			end
			S8:begin//Branch
			ALUSrcA=1;
			ALUSrcB=2'b00;
			ALUControl=3'b110;
			PCSrc=1;
			Branch=1;
			end
			default:;
		endcase
	end 
	
	always@(state or Op)
	begin
	case(state)
		S0:next_state=S1;
		S1:begin
		case(Op)
			LW:next_state=S2;
			SW:next_state=S2;
			R:next_state=S6;
			BEQ:next_state=S8;
		default:;
		endcase
		end
		S2:begin
		case(Op)
			LW:next_state=S3;
			SW:next_state=S5;
		default:;
		endcase
		end
		S3:next_state=S4;
		S4:next_state=S0;
		S5:next_state=S0;
		S6:next_state=S7;
		S7:next_state=S0;
		S8:next_state=S0;
		default:;
	endcase
	end

endmodule
