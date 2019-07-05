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

	//regfile
	reg rst;
	wire[5:0]rAddr,wAddr;
	wire[31:0]rDout,wDin;
	wire wEna;
	REG_FILE regfile(clk,rst,rAddr,rDout,wAddr,wDin,wEna);
	
	wire [5:0]rAddr_c,wAddr_c;
	wire [31:0]wDin_c;
	wire wEna_c;
	
	reg [5:0]rAddr_i,wAddr_i;
	reg [31:0]wDin_i;
	reg wEna_i;
	
	
	reg init;
	assign rAddr=init ? rAddr_c:rAddr_i;
	assign wAddr=init ? wAddr_c:wAddr_i;
	assign wDin=init ? wDin_c:wDin_i;
	assign wEna=init ? wEna_c:wEna_i;
	
	//ram
	wire [5:0]ADDRA,ADDRB;
	wire ENA,ENB,WEA;
	wire [31:0]DINA,DOUTB;
	
	RAM ram(.addra(ADDRA),.dina(DINA),.ena(ENA),
	.wea(WEA),.clka(clk),.addrb(ADDRB),.enb(ENB),.clkb(clk),.doutb(DOUTB));
	
	wire [5:0]ADDRA_c,ADDRB_c;
	wire [31:0]DINA_c;
	wire WEA_c;
	
	reg [5:0]ADDRA_i,ADDRB_i;
	reg [31:0]DINA_i;
	reg WEA_i;
	
	assign ENA=1;
	assign ENB=1;
	assign ADDRA=init ? ADDRA_c:ADDRA_i;
	assign ADDRB=init ? ADDRB_c:ADDRB_i;
	assign DINA=init ? DINA_c:DINA_i;
	assign WEA=init ? WEA_c:WEA_i;

	wire [31:0]alu_a,alu_b,alu_out;
	wire [4:0]alu_op;
	
	assign alu_op=1;
	
	Control control(clk,init,rDout,wDin_c,wEna_c,rAddr_c,wAddr_c,
					ADDRA_c,DINA_c,WEA_c,
					alu_a,alu_b,alu_out);
	ALU alu(alu_a,alu_b,alu_op,alu_out);
	
	initial
	begin
		init=0;
		wEna_i=1;
		WEA_i=0;
		//#2 ADDRA_i=0;DINA_i=2;
		//#2 ADDRA_i=1;DINA_i=2;
		#2 ADDRB_i=0;
		#2 wAddr_i=0;wDin_i=DOUTB;
		#2 ADDRA_i=1;
		#2 wAddr_i=1;wDin_i=DOUTB;
		#2 init=1;
	end
	
	always@(rAddr_c)
	begin
		if(rAddr_c==63)
		begin
			init=0;
			$finish;
		end
	end
endmodule