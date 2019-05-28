`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: USTC ESLAB（Embeded System Lab）
// Engineer: Haojun Xia
// Create Date: 2019/03/14 11:21:33
// Design Name: RISCV-Pipline CPU
// Module Name: NPC_Generator
// Target Devices: Nexys4
// Tool Versions: Vivado 2017.4.1
// Description: Choose Next PC value
//////////////////////////////////////////////////////////////////////////////////
module NPC_Generator(
    input wire [31:0] PCF,JalrTarget, BranchTarget, JalTarget,
    input wire BranchE,JalD,JalrE,
    output reg [31:0] PC_In,
	input wire PredF,
	input wire [1:0]Pred_Error,
	input wire [31:0]NPC_PredF,PCE
    );
    always @(*)
    begin//注意优先级
        if(JalrE)
            PC_In <= JalrTarget;			
        else if(Pred_Error[0])
            PC_In <= BranchTarget;
		else if(Pred_Error[1])
			PC_In <= PCE+4;
        else if(JalD)
            PC_In <= JalTarget;
		else if(PredF)
			PC_In <= NPC_PredF;
        else
            PC_In <= PCF+4;
    end
endmodule
