`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    19:40:20 04/04/2018 
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
	input clk,
	input init,
	input [31:0]rData,
	output reg [31:0]wData,
	output reg enable,
	output reg [5:0]rAddr,
	output reg [5:0]wAddr,
	
	output reg [5:0]ADDRA_c,
	output reg [31:0]DINA_c,
	output reg WEA_c,
	output reg [31:0]alu_a,
	output reg [31:0]alu_b,
	input wire [31:0]alu_out
	);
	
	localparam S1=2'b00;
	localparam S2=2'b01;
	localparam S3=2'b10;
	localparam S4=2'b11;
	
	
	
	
	
	
	reg [1:0]state=0;
	reg [1:0]next_state=0;
		
	initial
	begin
		enable=0;
		//WEA_c=0;
		rAddr=0;
		wAddr=0;
		//ADDRA_c=0;
		//alu_op=1;
	end
	
	always@(*)
	begin
		ADDRA_c=wAddr;
		WEA_c=enable;
		DINA_c=wData;
	end
	
	always@(posedge clk)
	begin
		if(init)
			state<=next_state;
	end 
	
	always@(state)
	begin
		case(state)
		S1:next_state=S2;
		S2:next_state=S3;
		S3:next_state=S4;
		S4:next_state=S1;
		default:next_state=S1;
		endcase
	end
	
	always@(state)
	begin
		case(state)
			S1:begin
				enable=0;
				//WEA_c=0;
				end
			S2:begin 
				alu_a=rData;
				rAddr=rAddr+1;
				end
			S3:begin 
				alu_b=rData;
				wAddr=rAddr+1;
				end
				
			S4:begin
				wData=alu_out;
				enable=1;
				end 
			default:;
		endcase
	end
endmodule
