`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    00:31:18 11/30/2017 
// Design Name: 
// Module Name:    cipher 
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
module cipher(clk,reset,key,flag,din,pos,dout);
input clk,reset;
input [4:0]key;
input flag;
input [5:0]din;
input [2:0]pos;
output reg[5:0]dout;

//assign dout=din;

reg [5:0]k1,k2,k3,k;
reg [6:0]temp;
//reg [5:0]sum;

reg S_delay=0;
wire S;
reg [5:0] count;
//always@(posedge clk)
//	count<=count+1;

always@(posedge clk)
	S_delay<=pos[0];

assign S=pos[0]^S_delay;

always@(posedge S,posedge reset)
	if(reset)
	begin
		k1={0,0,0,key[4],key[3],key[2]};
		k2={0,0,0,key[3],key[2],key[1]};
		k3={0,0,0,key[2],key[1],key[0]};
		dout=0;
	end
	else
	begin
		k1=k1+1;
		k2=k2+k1[2];
		k3=k3+k2[2]; 
		k=k1+k2+k3;
		if(k>35)
			k=k-36;
		
		if(flag)
		begin
			temp=din+k;
			if(temp>35)
				temp=temp-36;
			dout=temp[5:0];
		end
		else
		begin
			temp=din-k;
			if(din<k)
				temp=din+36-k;
			dout=temp[5:0];
		end
		//dout=pos;
//	if(dout>35)
//		dout=dout-36;
//	if(dout<0)
//		dout=dout+36;
	end       
	
//	always@(pos)
//		dout=din;
	
	
/*
always@(posedge reset,posedge S)
begin
	if(reset)
		dout=0;
	else
	begin
	dout=din+k1+k2+k3;
	if(dout>35)
		dout=dout-36;
	end
end

*/
endmodule