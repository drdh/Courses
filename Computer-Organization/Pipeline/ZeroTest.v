`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    15:07:06 05/12/2018 
// Design Name: 
// Module Name:    ZeroTest 
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
module ZeroTest(
	input [4:0]ALUCode,
	input [31:0]RsData,
	input [31:0]RtData,
	output reg Z
    );
localparam alu_beq=  5'b01010;     //0
localparam alu_bne=  5'b01011;  	//0
localparam alu_bgez= 5'b01100;     	//0
localparam alu_bgtz= 5'b01101;    //0
localparam alu_blez= 5'b01110;     //0
localparam alu_bltz= 5'b01111; 		//0
	
	always@(*)         
	begin           
	case(ALUCode)             
		alu_beq:Z<=&(RsData[31:0]~^RtData[31:0]);             
		alu_bne:Z<=|(RsData[31:0]^RtData[31:0]);             
		alu_bgez:Z<=~RsData[31];             
		alu_bgtz:Z<=~RsData[31]&&(|RsData[31: 0]);             
		alu_blez:Z<=RsData[31] || ~ (|RsData[31: 0]);             
		alu_bltz:Z<=RsData[31];             
	default: Z<=1'b0;           
	endcase         
	end 


endmodule
