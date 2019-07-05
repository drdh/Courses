`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:48:12 11/29/2017 
// Design Name: 
// Module Name:    keyboard 
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
module keyboard #(parameter W_SIZE=3)(clk,reset,ps2d,ps2c,data,pos,data_std);
input clk,reset;
input ps2d,ps2c;
output [5:0]data;
output [W_SIZE-1:0]pos;
output [5:0]data_std;

wire [7:0]key_code;
wire [7:0]r_data_std;
wire kb_not_empty,kb_buf_empty;
kb_code #(W_SIZE)kb_code_unit(.clk(clk),.reset(reset),
						.ps2d(ps2d),.ps2c(ps2c),
						.rd_key_code(kb_not_empty),.key_code(key_code),
						.kb_buf_empty(kb_buf_empty),.pos(pos),.r_data_std(r_data_std));

key2data key2data_unit(.key_code(key_code),.data(data));
key2data key2data_std(r_data_std,data_std);
assign kb_not_empty=~kb_buf_empty;
endmodule

//接收程序
module ps2_rx(clk,reset,ps2d,ps2c,rx_en,rx_done_tick,dout);
input clk,reset;
input ps2d,ps2c,rx_en;
output reg rx_done_tick;
output [7:0]dout;

//状态机信号
localparam [1:0]idle=2'b00,dps=2'b01,load=2'b10;

//信号声明
reg [1:0]state_reg,state_next;
reg [7:0]filter_reg;
wire [7:0]filter_next;
reg f_ps2c_reg;
wire f_ps2c_next;
reg [3:0]n_reg,n_next;
reg [10:0]b_reg,b_next;
wire fall_edge;

//主体
//滤波和下降沿检测ps2c
always@(posedge clk,posedge reset)
if(reset)
	begin
		filter_reg<=0;
		f_ps2c_reg<=0;
	end
else
	begin
		filter_reg<=filter_next;
		f_ps2c_reg<=f_ps2c_next;
	end
	
assign filter_next={ps2c,filter_reg[7:1]};
assign f_ps2c_next=(filter_reg==8'b11111111)?1'b1:(filter_reg==8'b00000000)?1'b0:f_ps2c_reg;
assign fall_edge=f_ps2c_reg&~f_ps2c_next;

//FSMD
//FSMD 状态和数据寄存器
always@(posedge clk,posedge reset)
	if(reset)
		begin
			state_reg<=idle;
			n_reg<=0;
			b_reg<=0;
		end
	else
		begin
			state_reg<=state_next;
			n_reg<=n_next;
			b_reg<=b_next;
		end

//FSMD下一状态逻辑
always@(*)
begin
	state_next=state_reg;
	rx_done_tick=1'b0;
	n_next=n_reg;
	b_next=b_reg;
	case(state_reg)
		idle:
			if(fall_edge&rx_en)
				begin
					//从开始位移位
					b_next={ps2d,b_reg[10:1]};
					n_next=4'b1001;
					state_next=dps;
				end
		dps:	//8位数据+1位校验+1位停止
			if(fall_edge)
				begin
					b_next={ps2d,b_reg[10:1]};
					if(n_reg==0)
						state_next=load;
					else
						n_next=n_reg-1;
				end
		load://额外一个时钟周期完成最后一位移位
			begin
				state_next=idle;
				rx_done_tick=1'b1;
			end
	endcase
end

//输出
assign dout=b_reg[8:1];	//数据位
endmodule

//PS/2 键盘通信接口电路
module kb_code #(parameter W_SIZE=3)(clk,reset,ps2d,ps2c,rd_key_code,key_code,kb_buf_empty,pos,r_data_std);
input clk,reset;
input ps2d,ps2c,rd_key_code;
output [7:0]key_code;
output kb_buf_empty;
output [W_SIZE-1:0]pos;
output [7:0]r_data_std;

//常数声明
localparam BRK=8'hf0;	//暂停编码
//状态定义
localparam wait_brk=1'b0,get_code=1'b1;

//信号声明
reg state_reg,state_next;
wire [7:0]scan_out;
reg got_code_tick;
wire scan_done_tick;

//主体
//例化ps/2接收器
ps2_rx ps2_rx_unit(.clk(clk),.reset(reset),
					.rx_en(1'b1),.ps2d(ps2d),
					.ps2c(ps2c),.rx_done_tick(scan_done_tick),
					.dout(scan_out));
//例化fifo缓冲器
fifo #(.B(8),.W(W_SIZE))
		fifo_key_unit(.clk(clk),.reset(reset),
						.rd(rd_key_code),.wr(got_code_tick),
						.w_data(scan_out),.empty(kb_buf_empty),
						.full(),.r_data(key_code),.pos(pos),.r_data_std(r_data_std));
						

//接收到F0时状态机获得按键扫描码
//状态寄存器
always@(posedge clk,posedge reset)
	if(reset)
		state_reg<=wait_brk;
	else
		state_reg<=state_next;

//下一状态逻辑
always@(*)
begin
	got_code_tick=1'b0;
	state_next=state_reg;
	case(state_reg)
		wait_brk:	//等待暂停码-F0
			if(scan_done_tick==1'b1&&scan_out==BRK)
				state_next=get_code;
		get_code:
			if(scan_done_tick)
				begin
					got_code_tick=1'b1;
					state_next=wait_brk;
				end
	endcase
end
endmodule
	
//ps/2译码程序
module key2data(key_code,data);
input [7:0]key_code;
output reg [5:0]data;
always@(*)
	case(key_code)
		8'h45:data=6'd0;
		8'h16:data=6'd1;
		8'h1e:data=6'd2;
		8'h26:data=6'd3;
		8'h25:data=6'd4;
		8'h2e:data=6'd5;
		8'h36:data=6'd6;
		8'h3d:data=6'd7;
		8'h3e:data=6'd8;
		8'h46:data=6'd9;
		
		8'h1c:data=6'd10;
		8'h32:data=6'd11;
		8'h21:data=6'd12;
		8'h23:data=6'd13;
		8'h24:data=6'd14;
		8'h2b:data=6'd15;
		8'h34:data=6'd16;
		8'h33:data=6'd17;
		8'h43:data=6'd18;
		8'h3b:data=6'd19;
		8'h42:data=6'd20;
		8'h4b:data=6'd21;
		8'h3a:data=6'd22;
		8'h31:data=6'd23;
		8'h44:data=6'd24;
		8'h4d:data=6'd25;
		8'h15:data=6'd26;
		8'h2d:data=6'd27;
		8'h1b:data=6'd28;
		8'h2c:data=6'd29;
		8'h3c:data=6'd30;
		8'h2a:data=6'd31;
		8'h1d:data=6'd32;
		8'h22:data=6'd33;
		8'h35:data=6'd34;
		8'h1a:data=6'd35;
	default:data=6'd0;
	endcase
endmodule