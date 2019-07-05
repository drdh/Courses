`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:46:00 04/04/2018 
// Design Name: 
// Module Name:    REG_FILE 
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
module REG_FILE(
	input clk,
	input rst,
	input[5:0] rAddr,
	output[31:0] rDout,
	input[5:0] wAddr,
	input[31:0] wDin,
	input wEna
    );
	reg [31:0]mem[0:63];
	
	always@(posedge clk,posedge rst)
	begin
		if(rst)
		begin:reset
			integer i;
			for(i=0;i<=63;i=i+1)
				mem[i]=0;
		end
		else if(wEna)
			mem[wAddr]=wDin;
		else
			mem[wAddr]=mem[wAddr];
	end
	
	assign rDout=wEna ? (rAddr==wAddr ? wDin : mem[rAddr]) : mem[rAddr];
endmodule
