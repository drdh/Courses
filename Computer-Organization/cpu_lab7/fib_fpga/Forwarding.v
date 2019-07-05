`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:16:20 05/12/2018 
// Design Name: 
// Module Name:    Forwarding 
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
module Forwarding(
	input RegWrite_wb,
	input RegWrite_mem,
	input [4:0]RegWriteAddr_wb,
	input [4:0]RegWriteAddr_mem,
	input [4:0]RsAddr_ex,
	input [4:0]RtAddr_ex,
	output [1:0]ForwardA,
	output [1:0]ForwardB
    );
	
	assign ForwardA[0]=RegWrite_wb&&(RegWriteAddr_wb!=0)&&(!((RegWriteAddr_mem==RsAddr_ex)&&(RegWrite_mem)))&&(RegWriteAddr_wb==RsAddr_ex);   
	assign ForwardA[1]=RegWrite_mem&&(RegWriteAddr_mem!=0)&&(RegWriteAddr_mem==RsAddr_ex);   
	assign ForwardB[0]=RegWrite_wb&&(RegWriteAddr_wb!=0)&&(!((RegWriteAddr_mem==RtAddr_ex)&&(RegWrite_mem)))&&(RegWriteAddr_wb==RtAddr_ex);   
	assign ForwardB[1]=RegWrite_mem&&(RegWriteAddr_mem!=0)&&(RegWriteAddr_mem==RtAddr_ex); 
	
endmodule
