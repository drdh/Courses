`timescale 1ns / 1ps

module BHT #(
	parameter TABLE_LEN=4
)(
	input wire clk,
	input wire rst,
	input wire [31:0]PCF,
	output reg PredF,
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
	//reg Extra_Bit[TABLE_SIZE];//表示有效
	reg[1:0]State_Buff[TABLE_SIZE];
	//SN:00	WN:01	WT:10 ST:11
	
	wire [TABLE_LEN-1:0] Pred_PC_in=PCF[TABLE_LEN+1:2];
	wire [TABLE_LEN-1:0] Update_PC_in=PCE[TABLE_LEN+1:2];
	
	wire Hit_Buff_Pred=(Target_Buff_Tag[Pred_PC_in]==PCF);
	wire Hit_Buff_Update=(Target_Buff_Tag[Update_PC_in]==PCE);
	
	assign NPC_PredF=Target_Buff[Pred_PC_in];
	//assign PredF=State_Buff[Pred_PC_in][1]==1'b1 ? //需要BHT的预测
	//	(Extra_Bit[Pred_PC_in] && Hit_Buff_Pred) : 1'b0;
	
	always@(*)
		//if(State_Buff[Pred_PC_in][1]==1'b1 && (Extra_Bit[Pred_PC_in] && Hit_Buff_Pred))
		if(State_Buff[Pred_PC_in][1]==1'b1 && Hit_Buff_Pred)
			PredF<=1'b1;
		else 
			PredF<=1'b0;
	
	integer i;
	always @ (negedge clk or posedge rst)
	begin
		if(rst)begin
			for(i=0;i<TABLE_SIZE;i=i+1)begin
				Target_Buff[i]<=0;
				Target_Buff_Tag[i]<=0;
				//Extra_Bit[i]<=0;
				State_Buff[i]<=2'b00;
			end
		end
		else if(BranchE)begin
			Target_Buff[Update_PC_in]<=BrNPC;
			Target_Buff_Tag[Update_PC_in]<=PCE;
			//Extra_Bit[Update_PC_in]<=1'b1;
			
			if(Hit_Buff_Update)begin
				case(State_Buff[Update_PC_in])
					2'b00:State_Buff[Update_PC_in]<=2'b01;
					2'b01:State_Buff[Update_PC_in]<=2'b11;
					2'b10:State_Buff[Update_PC_in]<=2'b11;
					2'b11:State_Buff[Update_PC_in]<=2'b11;
					default:State_Buff[Update_PC_in]<=2'b11;
				endcase
			end
			else begin
				State_Buff[Update_PC_in]<=2'b10;
			end
			
		end
		else if((!BranchE) && Hit_Buff_Update)begin//实际没有taken,且预测单元有此项
			//Extra_Bit[Update_PC_in]<=1'b0;
			case(State_Buff[Update_PC_in])
				2'b00:State_Buff[Update_PC_in]<=2'b00;
				2'b01:State_Buff[Update_PC_in]<=2'b00;
				2'b10:State_Buff[Update_PC_in]<=2'b00;
				2'b11:State_Buff[Update_PC_in]<=2'b10;
				default:State_Buff[Update_PC_in]<=2'b00;
			endcase
		end
	end
endmodule
