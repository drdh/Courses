`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    15:16:29 05/12/2018 
// Design Name: 
// Module Name:    HazardDetector 
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
module HazardDetector(
	input [4:0]RegWriteAddr,
	input MemRead,
	input [4:0]RsAddr,
	input [4:0]RtAddr,
	output Stall,
	output PC_IFWrite
    );
assign Stall=MemRead&&((RegWriteAddr==RsAddr)||(RegWriteAddr==RtAddr));   
assign PC_IFWrite = ~Stall; 


endmodule
