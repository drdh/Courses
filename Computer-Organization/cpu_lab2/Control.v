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
	output reg [5:0]wAddr);
	
	localparam S1=2'b00;	//4个状态
	localparam S2=2'b01;
	localparam S3=2'b10;
	localparam S4=2'b11;
	
	reg[31:0]alu_a,alu_b;
	reg[4:0]alu_op;
	wire [31:0]alu_out;
	
	ALU alu(alu_a,alu_b,alu_op,alu_out);	//alu模块表示加法
	
	
	reg [1:0]state=0;
	reg [1:0]next_state=0;
		
	initial
	begin
		enable=0;
		rAddr=0;
		wAddr=0;
		alu_op=1;
	end
		
	always@(posedge clk)
	begin
		if(init)	//当init=1，表示初始化结束，才开始改变状态
			state<=next_state;
	end 
	
	always@(state)
	begin
		case(state)	//状态转移时循环的
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
			S1:enable=0;
			S2:begin 
				alu_a=rData;	//读出第一个数
				rAddr=rAddr+1;	//为读第二个数改变地址
				end
			S3:begin 	
				alu_b=rData;	//读第二个数
				wAddr=rAddr+1;	//为写入改变地址
				end
				
			S4:begin
				wData=alu_out;	//写入数据是Alu自行计算的
				enable=1;
				end 
			default:;
		endcase
	end
endmodule
