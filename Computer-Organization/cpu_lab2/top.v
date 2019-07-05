`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   19:06:59 04/04/2018
// Design Name:   REG_FILE
// Module Name:   C:/Users/drdh/Documents/ISE/lab2/lab2/top.v
// Project Name:  lab2
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: REG_FILE
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module top;
	reg clk;
	
	initial
	begin
		clk=0;
		$monitor($time,"       %d",wAddr);
		forever #1 clk=~clk;
	end 

	reg rst;
	wire[5:0]rAddr,wAddr;	//以下，寄存器文件的接口
	wire[31:0]rDout,wDin;
	wire wEna;
	REG_FILE regfile(clk,rst,rAddr,rDout,wAddr,wDin,wEna);
	
	
	wire [5:0]rAddr_c,wAddr_c;	//c表示control，表示和control的接口
	wire [31:0]wDin_c;
	wire wEna_c;
	
	reg [5:0]rAddr_i,wAddr_i;	//i表示init，初始时的接口
	reg [31:0]wDin_i;
	reg wEna_i;
	
	
	reg init;	//当init=0时，状态机不改变状态，而且，寄存器文件的接口，使用i
	assign rAddr=init ? rAddr_c:rAddr_i;
	assign wAddr=init ? wAddr_c:wAddr_i;
	assign wDin=init ? wDin_c:wDin_i;
	assign wEna=init ? wEna_c:wEna_i;
	
	Control control(clk,init,rDout,wDin_c,wEna_c,rAddr_c,wAddr_c);
	
	initial
	begin	//初始化
		init=0;
		wEna_i=1;
		#10 wAddr_i=0;wDin_i=2;
		#10 wAddr_i=1;wDin_i=2;
		#10 init=1;	//control状态机开始改变状态
	end
	
	always@(rAddr_c)	//结束仿真标志
	begin
		if(rAddr_c==63)
			$finish;
	end
endmodule