`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    16:43:29 11/29/2017 
// Design Name: 
// Module Name:    fifo 
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
module fifo #(parameter B=8,W=4)(clk,reset,rd,wr,w_data,empty,full,r_data,pos,r_data_std);		//数据位宽，地址位宽
input clk,reset;
input rd,wr;
input [B-1:0]w_data;
output empty,full;
output [B-1:0]r_data;
output [W-1:0]pos;
output [B-1:0]r_data_std;



reg [B-1:0]array_reg [2**W-1:0];	//寄存器数组
reg [W-1:0]w_ptr_reg,w_ptr_next,w_ptr_succ;
reg [W-1:0]r_ptr_reg,r_ptr_next,r_ptr_succ;
reg full_reg,empty_reg,full_next,empty_next;
wire wr_en;

assign pos=r_ptr_reg;
//主体部分
//寄存器文件写操作
always@(posedge clk)
	if(wr_en)
		array_reg[w_ptr_reg]<=w_data;
		
//寄存器文件读操作
assign r_data=array_reg[r_ptr_reg-1];
assign r_data_std=array_reg[r_ptr_reg];
//不满的时候写使能有效
assign wr_en=wr&~full_reg;	//wr是接口，表示需要写

//FIFO控制逻辑
//寄存器读写指针
always@(posedge clk,posedge reset)
	if(reset)
		begin
			w_ptr_reg<=0;
			r_ptr_reg<=0;
			//w_ptr_next<=0;
			//w_ptr_succ<=0;
			//r_ptr_next<=0;
			//r_ptr_succ<=0;
			full_reg<=1'b0;
			empty_reg<=1'b1;
		end
	else
		begin
			w_ptr_reg<=w_ptr_next;
			r_ptr_reg<=r_ptr_next;
			full_reg<=full_next;
			empty_reg<=empty_next;
		end
	
//读写指针的下一状态逻辑
//integer i;
always@(*)
begin
	//指针加1操作
	w_ptr_succ=w_ptr_reg+1;
	r_ptr_succ=r_ptr_reg+1;
	//默认保持原值
	w_ptr_next=w_ptr_reg;
	r_ptr_next=r_ptr_reg;
	full_next=full_reg;
	empty_next=empty_reg;
	
	case({wr,rd})
		2'b01:	//读操作
			if(~empty_reg)	//非空
				begin
					r_ptr_next=r_ptr_succ;
					full_next=1'b0;
					if(r_ptr_succ==w_ptr_reg)
						empty_next=1'b1;
				end
		2'b10:	//写操作
			if(~full_reg)	//非满
				begin
					w_ptr_next=w_ptr_succ;
					empty_next=1'b0;
					if(w_ptr_succ==r_ptr_reg)
						full_next=1'b1;
				end
		2'b11:	//读和写
			begin
				w_ptr_next=w_ptr_succ;
				r_ptr_next=r_ptr_succ;
			end
		//default:i=1;
	endcase
end

//输出
assign full=full_reg;
assign empty=empty_reg;
endmodule

