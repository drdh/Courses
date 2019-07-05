`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    05:22:02 05/12/2018 
// Design Name: 
// Module Name:    Registers 
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
module Registers(
	input clk,
	input [4:0]RsAddr,
	input [4:0]RtAddr,
	input [31:0]WriteData,
	input [4:0]WriteAddr,
	input RegWrite,
	output [31:0]RsData,
	output [31:0]RtData
    );
	reg [31:0]Regs[0:31];
	
	assign RsData=(RsAddr==5'b0) ? 32'b0 : Regs[RsAddr];
	assign RtData=(RtAddr==5'b0) ? 32'b0 : Regs[RtAddr];
	
	always@(posedge clk) 
	begin  
		if(RegWrite)    
			Regs[WriteAddr]<=WriteData;  
	end 

endmodule
