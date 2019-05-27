`timescale 1ns / 1ps
module BTB #(
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
	reg Extra_Bit[TABLE_SIZE];//表示上次采用
	
	wire [TABLE_LEN-1:0] Pred_PC_in=PCF[TABLE_LEN+1:2];
	wire [TABLE_LEN-1:0] Update_PC_in=PCE[TABLE_LEN+1:2];
	
	assign NPC_PredF=Target_Buff[Pred_PC_in];
	assign PredF=(Extra_Bit[Pred_PC_in] && Target_Buff_Tag[Pred_PC_in]==PCF);
	
	integer i;
	always @(negedge clk or posedge rst)
	begin
		if(rst)begin
			for(i=0;i<TABLE_SIZE;i=i+1)begin
				Target_Buff[i]<=0;
				Target_Buff_Tag[i]<=0;
				Extra_Bit[i]<=0;
			end
		end
		else if(BranchE)begin//简化code,实际taken,不管是否正确预测,由于是直接相联,均更新
			Target_Buff[Update_PC_in]<=BrNPC;
			Target_Buff_Tag[Update_PC_in]<=PCE;
			Extra_Bit[Update_PC_in]<=1'b1;
		end
		else if((!BranchE)&& PredE && (Target_Buff_Tag[Update_PC_in]==PCE))begin//预测Taken, 实际未taken,如果有此项，则置零
			Extra_Bit[Update_PC_in]<=1'b0;
		end
	end

endmodule
