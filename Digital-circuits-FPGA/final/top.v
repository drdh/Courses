`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    12:30:02 11/29/2017 
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

module top(clk,reset,ps2c,ps2d,key,flag,off,seg,an,led,lcd_en,lcd_rw,lcd_rs,lcd_data,audio);
	input clk,reset;
	//input [5:0]data;
	input ps2c,ps2d;
	input [4:0]key;
	input flag;
	input off;
	output [7:0]seg;
	output [3:0]an;
	output [7:0]led;
	output lcd_en,lcd_rw,lcd_rs;
	output [7:0]lcd_data;
	output audio;
	
	wire [5:0]data;
	wire [4:0]code;	
	wire [2:0]width;	//code width
	wire [2:0]pos;
	wire [5:0]dout;
	wire [5:0]data_std;
	
	keyboard #(2)Keyboard(clk,reset,ps2d,ps2c,data,pos,data_std);
	cipher Cipher(clk,reset,key,flag,data_std,pos,dout);
	
	
	to_morse	toMorse(dout,code,width);
	
	
	display Display(clk,data,dout,seg,an);
	ledplay ledPlay(clk,code,width,led);
	lcd lcdDisp(clk,reset,pos,data,dout,lcd_en,lcd_rw ,lcd_rs ,lcd_data);
	kenan_player player(clk,off,audio);
endmodule
