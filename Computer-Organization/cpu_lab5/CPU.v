`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:29:42 04/15/2018 
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
	#5;
	forever #10 clk=~clk;
	end
	
	//control
	wire [31:0]Instr;
	wire RegWrite,RegDst,ALUSrc,Branch,MemWrite,MemtoReg,Jump,bgtz;
	wire [2:0]ALUControl;
	Control_Unit CONTROL(Instr[31:26],Instr[5:0],RegWrite,RegDst,ALUSrc,ALUControl,Branch,MemWrite,MemtoReg,Jump,bgtz);
	
	//PC
	wire PCSrc;
	wire [31:0]PCBranch,PCPlus4,PC_,PC,PCJump,PCSrcA;
	assign PCSrcA=PCSrc ? PCBranch : PCPlus4;
	assign PC_=Jump ? PCJump : PCSrcA;
	
	PC PC_unit(clk,PC_,PC);
	
	//instruction memory
	//[5:0]PC
	assign PCPlus4=PC+4;
	Instruction_Memory Instr_unit(PC,Instr);
	
	//register file
	wire [4:0]WriteReg;
	wire [31:0]Result,RD1,RD2;
	wire [4:0]A1,A2;
	assign A1=Instr[25:21];
	assign A2=Instr[20:16];
	Register_File reg_file(clk,A1,A2,WriteReg,Result,RegWrite,RD1,RD2);
	//Register_File reg_file(clk,Instr[25:21],Instr[20:16],WriteReg,Result,RegWrite,RD1,RD2);
	
	assign WriteReg=RegDst ? Instr[15:11] : Instr[20:16];
	
	//sign extend
	wire [31:0]SignImm;
	assign SignImm[15:0] = Instr[15:0];  
   assign SignImm[31:16] = Instr[15]? 16'hffff : 16'h0000;
	//assign SignImm[31:0]={16{Instr[15]},Instr[15:0]};
	
	//Jump
	assign PCJump=(Instr[25:0]<<2);
	
	//ALU
	wire [31:0]SrcA,SrcB,ALUResult;
	wire Zero;
	assign SrcA=RD1;
	assign SrcB=ALUSrc ? SignImm : RD2;
	ALU ALU_unit(ALUControl,SrcA,SrcB,ALUResult,Zero);
	
	//branch & bgtz
	//assign PCSrc=Zero & Branch;
	assign PCSrc=(Zero & Branch) | ((~SrcA[31]) & (| SrcA[30:0]) & bgtz);
	assign PCBranch=(SignImm<<2)+PCPlus4;
	
	//data memory
	wire [31:0]WriteData,ReadData;
	assign WriteData=RD2;
	Data_Memory data_unit(clk,MemWrite,ALUResult,WriteData,ReadData);
	
	assign Result=MemtoReg ? ReadData : ALUResult;
	
/*	always@(Instr)
	begin
		if(Instr==32'h08000011)
			$finish;
	end
*/																																		
endmodule
