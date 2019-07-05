`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:09:27 11/15/2017 
// Design Name: 
// Module Name:    top 
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
module top(clk,rst_n,enable,seq,led);
	 input clk,rst_n,enable,seq;
	 output led;
	 
	 wire key;
	 wire clk_div;
	 clkdiv U1(clk,clk_div);
	 debounce U2(clk_div,enable,key);
	 detect U3(clk,rst_n,key,seq,led);

endmodule


module clkdiv(clk,clk_div);
    input clk;
    output clk_div;
	 reg [17:0]count=0;
	 
	 always @(posedge clk)
	     count<=count+1;
	 assign clk_div=count[17];
endmodule

module debounce(clk_div,en,key);
    input clk_div,en;
	 output key;
	 reg k1=0,k2=0,k3=0;
	 always @(posedge clk_div)
	 begin
	     k3=k2;
		  k2=k1;
		  k1=en;
	 end
	 assign key=k1&k2&k3;
endmodule

module detect(clk,rst_n,key,seq,led);
    input clk,rst_n,key,seq;
	 output reg led;
	 
	 localparam [2:0] 
	 s0=3'b000,
	 s1=3'b001,
	 s2=3'b010,
	 s3=3'b011,
	 s4=3'b100;
	 
	 reg [2:0] cur_state,next_state;
	 
	 always@(posedge clk or negedge rst_n)
	     if(~rst_n)
		      cur_state<=s0;
	     else
		      cur_state<=next_state;
	
	 always@(posedge key or negedge rst_n)
	 begin
	     if(~rst_n)
		      next_state=s0;
		  else
		  begin
	     case(cur_state)
		      s0: if(seq)
				        next_state=s1;
					 else
					     next_state=s0;
				s1: if(seq)
				        next_state=s2;
					 else
					     next_state=s0;
				s2: if(seq)
				        next_state=s2;
					 else
					     next_state=s3;
				s3: if(seq)
				        next_state=s4;
					 else
					     next_state=s0;
				s4: if(seq)
				        next_state=s1;
					 else
					     next_state=s0;
				default:next_state=s0;
			endcase
			end
		end
		
		always@(*)
		    if(cur_state==s4)
			     led=1'b1;
			 else
			     led=1'b0;
endmodule
	     
	 
	 
	 

