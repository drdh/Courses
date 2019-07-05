`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:59:27 11/01/2017 
// Design Name: 
// Module Name:    compare4 
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
module compare4(a,b,oa,oe,ob
    );
	 input [3:0] a,b;
	 output oa,oe,ob;
	 
	 assign oa=(a>b)? 1'b1:1'b0;
	 assign oe=(a==b)? 1'b1:1'b0;
	 assign ob=(a<b)? 1'b1:1'b0;


endmodule

