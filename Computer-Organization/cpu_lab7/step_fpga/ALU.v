`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    13:42:30 05/12/2018 
// Design Name: 
// Module Name:    ALU 
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
module ALU(
	input [4:0]ALUCode,
	input [31:0]A,
	input [31:0]B,
	output reg[31:0]Result
    );
	
/*	reg signed[31:0]B_reg;
	always@(B)
	begin
		B_reg=B;
	end 
*/
	
localparam alu_add=  5'b00000;    
localparam alu_and=  5'b00001;    
localparam alu_xor=  5'b00010;   
localparam alu_or =  5'b00011;    
localparam alu_nor=  5'b00100;    
localparam alu_sub=  5'b00101;    
localparam alu_andi= 5'b00110; 
localparam alu_xori= 5'b00111; 
localparam alu_ori = 5'b01000; 
localparam alu_jr =  5'b01001; 
localparam alu_beq=  5'b01010;    
localparam alu_bne=  5'b01011; 
localparam alu_bgez= 5'b01100;    
localparam alu_bgtz= 5'b01101;   
localparam alu_blez= 5'b01110;    
localparam alu_bltz= 5'b01111; 
localparam alu_sll=  5'b10000; 
localparam alu_srl=  5'b10001; 
localparam alu_sra=  5'b10010;  
localparam alu_slt=  5'b10011;    
localparam alu_sltu= 5'b10100; 
localparam alu_addu= 5'b10101;
localparam alu_subu= 5'b10110;

	always@(*)     
	begin       
	case(ALUCode)         
		alu_add  :Result<=$signed(A)+$signed(B); 
		alu_addu :Result<=A+B;
		alu_and  :Result<=A&B;         
		alu_xor  :Result<=A^B;         
		alu_or   :Result<=A|B;         
		alu_nor  :Result<=~(A|B);         
		alu_sub  :Result<=$signed(A)-$signed(B); 
		alu_subu :Result<=A-B;
		alu_andi :Result<=A&{16'd0,B[15:0]};    //以下使用了sign extend  
		alu_xori :Result<=A^{16'd0,B[15:0]};    //但实际上是zero extend 
		alu_ori  :Result<=A|{16'd0,B[15:0]};    //从而在ALU中改正
		alu_sll  :Result<=B<<A;      
		alu_srl  :Result<=B>>A;      
		//alu_sra  :Result<=B_reg>>>A; 
		alu_sra  :Result<=$signed(B) >>>A;
		//alu_slt  :Result<=(A[31]&&(~B[31]))||((A[31]^~B[31])&&sum[31]);     
		alu_slt	 :Result<=($signed(A) < $signed(B)) ? 1 : 0;
		//alu_sltu :Result<=((~A[31])&&B[31])||((A[31]~^B[31])&&sum[31]);  
		alu_sltu :Result<=(A<B)?1:0;
		default  :Result<=32'b0;       
	endcase     
	end 
	
endmodule
