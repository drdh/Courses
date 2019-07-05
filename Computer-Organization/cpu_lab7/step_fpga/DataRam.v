`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:57:36 05/12/2018 
// Design Name: 
// Module Name:    DataRam 
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
module DataRam(
	input clk,
	input [31:0]addr,
	input [31:0]din,
	input we,
	output [31:0]dout
    );
	//reg [31:0]Mem[0:63];
	
	wire [4:0]A;
	assign A=(addr>>2);
	RAM Mem(.a(A),.d(din),.we(we),.clk(clk),.spo(dout));
	//RAM_BlOCK Mem(.addra(A),.dina(din),.wea(we),.clka(clk),.douta(dout));
	//assign dout=Mem[A];
	
/*	always@(posedge clk)
	begin
		//dout<=Mem[A];
		if(we)
			Mem[A]<=din;
	end 
*/
/*	
	initial
	begin
	Mem[0]=32'h00000000;
	
	
	Mem[20]=32'h00000014;
	Mem[21]=32'h00000003;
	Mem[22]=32'h00000003;
	
	end
*/	
endmodule
