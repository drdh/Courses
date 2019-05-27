`timescale 1ns / 1ps

module BHT #(
	parameter TABLE_LEN=4
)(
	input wire clk,
	input wire rst,
	input wire [31:0]PCF,
	output wire PredF,
	output wire [31:0]NPC_PredF,
	
	input wire [31:0]PCE,
	input wire PredE,
	input wire BranchE,
	input wire [31:0]NPC_PredE,
	input wire [31:0]BrNPC
);
	localparam TABLE_SIZE=1<<TABLE_LEN;
	reg [31:0]Target_Buff[TABLE_SIZE];
	reg [31:0]Target_Buff_Tag[TABLE_SIZE];
	reg Extra_Bit[TABLE_SIZE];//表示有效
	reg [1:0]State_Buff[TABLE_SIZE];
	//SN:00	WN:01	WT:10 ST:11
	
	wire [TABLE_LEN-1:0] Pred_PC_in=PCF[TABLE_LEN+1:2];
	wire [TABLE_LEN-1:0] Update_PC_in=PCE[TABLE_LEN+1:2];
	
	assign NPC_PredF=Target_Buff[Pred_PC_in];
	assign PredF=State_Buff[Pred_PC_in][1]==1'b1 ? 
		(Extra_Bit[Pred_PC_in] && Target_Buff_Tag[Pred_PC_in]==PCF) : 1'b0;
	
	
	


endmodule
