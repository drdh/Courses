`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    12:21:47 04/18/2018 
// Design Name: 
// Module Name:    Control 
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
module Control(
	input clk,
	input rst,
	output [31:0]out);
	
	reg [7:0]ramWtAddr,ramRdAddr;
	wire [31:0]ramRdOut;
	wire [31:0]ramWtIn;
	reg ramWtEn;
	RAM ram(.addra(ramWtAddr),.dina(ramWtIn),.wea(ramWtEn),.clka(clk),
			.addrb(ramRdAddr),.clkb(~clk),.doutb(ramRdOut));
	
	reg [5:0]fileRdAddr,fileWtAddr;
	wire [31:0]fileWtIn;
	wire [31:0]fileRdOut;
	reg fileWtEn;
	REG_FILE reg_file(clk,rst,fileRdAddr,fileRdOut,fileWtAddr,fileWtIn,fileWtEn);
	
	reg [31:0]alu_a,alu_b;
	wire [31:0]alu_out;
	reg [4:0]alu_op;
	ALU alu_unit(alu_a,alu_b,alu_op,alu_out);
	
	assign fileWtIn=ramRdOut;
	assign ramWtIn=alu_out;
	assign out=ramRdOut;
	reg [7:0]read1,read2,write;
	
	localparam S0=2'b00;
	localparam S1=2'b01;
	localparam S2=2'b10;
	localparam S3=2'b11;
	
	reg [1:0]state,next_state;
	
	always@(posedge clk,posedge rst)
	begin
		if(rst)
			state<=S0;
		else
			state<=next_state;	
	end
	
	always@(*)
	begin
		case(state)
		S0:next_state=S1;
		S1:next_state=S2;
		S2:next_state=S3;
		S3:next_state=S1;
		default:next_state=S0;
		endcase
	end

	
	always@(posedge clk,posedge rst)
	begin
		if(rst)
		begin
			read1<=0;
			read2<=8'd100;
			write<=8'd200;
			
			ramRdAddr<=0;
			ramWtAddr<=0;
			ramWtEn<=0;

			fileRdAddr<=0;
			fileWtAddr<=0;
			fileWtEn<=0;
		end 
		else
		begin
			case(next_state)
			S0:begin
			read1<=0;
			read2<=8'd100;
			write<=8'd200;
			
			ramRdAddr<=0;
			ramWtAddr<=8'd200;
			ramWtEn<=0;
			
			fileRdAddr<=0;
			fileWtAddr<=0;
			fileWtEn<=1;
			end
			S1:begin
			read1<=read1+1;
			ramRdAddr<=read1+1;
			alu_a<=ramRdOut;
			ramWtEn<=0;
			
			fileWtAddr<=1;
			fileWtEn<=1;
			
			
			end
			S2:begin
			read1<=read1+1;
			ramRdAddr<=read2;
			alu_b<=ramRdOut;
			
			fileWtAddr<=2;
			fileWtEn<=1;

			end
			S3:begin
			read2<=read2+1;
			ramRdAddr<=read1;
			alu_op<=ramRdOut;
			write<=write+1;
			ramWtAddr<=write;
			ramWtEn<=1;
			
			fileWtAddr<=0;
			fileWtEn<=1;
			
			end
			default:;
			endcase
		end
	end
endmodule
