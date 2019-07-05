`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:22:57 04/16/2018 
// Design Name: 
// Module Name:    CPU 
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
module CPU;
	//clk
	reg clk;
	initial
	begin
	clk=0;
	#30;
	forever #10 clk=~clk;
	end
	
	//control
	wire [31:0]Instr;
	wire RegWrite,ALUSrcA,PCSrc,Branch,PCWrite,IorD,MemWrite,IRWrite,RegDst,MemtoReg;
	wire [1:0]ALUSrcB;
	wire [2:0]ALUControl;
	Control control_unit(clk,Instr[31:26],Instr[5:0],RegWrite,ALUSrcA,ALUSrcB,ALUControl,PCSrc,Branch,PCWrite,IorD,MemWrite,IRWrite,RegDst,MemtoReg);
	
	//PC
	wire [5:0]PC_,PC;
	wire PCEn;
	Register #(6)PC_reg(clk,PCEn,PC_,PC);
	
	//Memory
	wire [5:0]Addr;
	wire [31:0]B,RD,ALUOut;
	assign Addr=IorD ? ALUOut : PC;
	Memory memory_unit(clk,MemWrite,Addr,B,RD);
	
	//Instr
	Register #(32)Instr_reg(clk,IRWrite,RD,Instr);
	
	//Data
	wire [31:0]Data;
	Register #(32)Data_reg(clk,1'b1,RD,Data);
	
	//register file
	wire [4:0]A1,A2,A3;
	wire [31:0]WD3,RD1,RD2;
	assign A1=Instr[25:21];
	assign A2=Instr[20:16];
	assign A3=RegDst ? Instr[15:11] : Instr[20:16];
	assign WD3=MemtoReg ? Data : ALUOut;
	Register_File reg_file(clk,A1,A2,A3,WD3,RegWrite,RD1,RD2);
	
	//sign extend
	wire [31:0]SignImm;
	assign SignImm[15:0] = Instr[15:0];  
    assign SignImm[31:16] = Instr[15]? 16'hffff : 16'h0000;
	
	//Read register A,B reg
	wire [31:0]A;
	Register #(32)A_reg(clk,1'b1,RD1,A);
	Register #(32)B_reg(clk,1'b1,RD2,B);
	
	//ALU 
	wire [31:0]SrcA,SrcB;
	assign SrcA=ALUSrcA ? A : PC;
	Mux4 #(32)mux_SrcB(ALUSrcB,B,4,SignImm,(SignImm<<2),SrcB);
	wire Zero;
	wire [31:0]ALUResult;
	ALU ALU_unit(ALUControl,SrcA,SrcB,ALUResult,Zero);
	
	Register #(32)ALUOut_reg(clk,1'b1,ALUResult,ALUOut);
	
	//PC [5:0]; ALU [31:0] 
	assign PC_=PCSrc ? ALUOut[5:0] : ALUResult[5:0];
	
	//branch
	assign PCEn=(Branch & Zero) | PCWrite;
	
	always@(PC)
	begin
		if(PC==24)	
		//if(!Instr)
		$finish;
	end

endmodule
