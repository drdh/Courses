`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    10:51:30 05/12/2018 
// Design Name: 
// Module Name:    Decode 
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
module Decode(
	 input [31:0]Instruction,
	 output MemtoReg,
	 output RegWrite,
	 output MemWrite,
	 output MemRead,
	 output reg[4:0] ALUCode,
	 output ALUSrcA,
	 output ALUSrcB,
	 output RegDst,
	 output J,
	 output JR
    );
//Instruction Field
wire [5:0] op;
wire [4:0] rt;
wire [5:0]funct;
assign op=Instruction[31:26];
assign rt=Instruction[20:16];
assign funct=Instruction[5:0];


//R_type
localparam R_type_op = 6'b000000;     
localparam ADD_funct = 6'b100000;  
localparam ADDU_funct= 6'b100001;  
localparam AND_funct = 6'b100100;  
localparam XOR_funct = 6'b100110;  
localparam OR_funct  = 6'b100101;  
localparam NOR_funct = 6'b100111;  
localparam SUB_funct = 6'b100010;  
localparam SUBU_funct= 6'b100011; 
localparam SLT_funct = 6'b101010;  
localparam SLTU_funct= 6'b101011;  
localparam SLL_funct = 6'b000000; 
localparam SLLV_funct= 6'b000100;  
localparam SRL_funct = 6'b000010;  
localparam SRLV_funct= 6'b000110;  
localparam SRA_funct = 6'b000011;  
localparam SRAV_funct= 6'b000111;  
localparam JR_funct  = 6'b001000; 
//R_type1
wire ADD,ADDU,AND,NOR,OR,SLT,SLTU,SUB,SUBU,XOR,SLLV, SRAV, SRLV,R_type1; 
assign ADD=(op==R_type_op)&&(funct==ADD_funct);  
assign ADDU=(op==R_type_op)&&(funct==ADDU_funct);  
assign AND=(op==R_type_op)&&(funct==AND_funct);  
assign NOR=(op==R_type_op)&&(funct==NOR_funct);  
assign OR=(op==R_type_op)&&(funct==OR_funct);  
assign SLT=(op==R_type_op)&&(funct==SLT_funct);  
assign SLTU=(op==R_type_op)&&(funct==SLTU_funct);  
assign SUB=(op==R_type_op)&&(funct==SUB_funct);  
assign SUBU=(op==R_type_op)&&(funct==SUBU_funct);  
assign XOR=(op==R_type_op)&&(funct==XOR_funct);  
assign SLLV=(op==R_type_op)&&(funct==SLLV_funct);  
assign SRAV=(op==R_type_op)&&(funct==SRAV_funct);  
assign SRLV=(op==R_type_op)&&(funct==SRLV_funct);
assign R_type1=ADD||ADDU||AND||NOR||OR||SLT||SLTU||SUB||SUBU||XOR||SLLV||SRAV||SRLV;
//R_type2
wire SLL,SRA,SRL,R_type2;   
assign SLL=(op==R_type_op)&&(funct==SLL_funct)&&(|Instruction);  //巨坑，与nop的区别
assign SRA=(op==R_type_op)&&(funct==SRA_funct);  
assign SRL=(op==R_type_op)&&(funct==SRL_funct);  
assign R_type2=SLL||SRA||SRL;


//Branch
localparam BEQ_op = 6'b000100; 
localparam BNE_op = 6'b000101; 
localparam BGEZ_op= 6'b000001;  
localparam BGEZ_rt= 5'b00001;  
localparam BGTZ_op= 6'b000111;  
localparam BGTZ_rt= 5'b00000;  
localparam BLEZ_op= 6'b000110;  
localparam BLEZ_rt= 5'b00000;  
localparam BLTZ_op= 6'b000001;  
localparam BLTZ_rt= 5'b00000; 

wire BEQ,BGEZ,BGTZ,BLEZ,BLTZ,BNE,Branch; 
assign BEQ=(op==BEQ_op);  
assign BNE=(op==BNE_op);  
assign BGEZ=(op==BGEZ_op)&&(rt==BGEZ_rt);  
assign BGTZ=(op==BGTZ_op)&&(rt==BGTZ_rt);  
assign BLEZ=(op==BLEZ_op)&&(rt==BLEZ_rt);  
assign BLTZ=(op==BLTZ_op)&&(rt==BLTZ_rt);  
assign Branch=BEQ||BNE||BGEZ||BGTZ||BLEZ||BLTZ;


//I 
localparam ADDI_op = 6'b001000;  
localparam ADDIU_op= 6'b001001; 
localparam ANDI_op = 6'b001100;  
localparam XORI_op = 6'b001110;  
localparam ORI_op  = 6'b001101;  
localparam SLTI_op = 6'b001010; 
localparam SLTIU_op= 6'b001011;    
wire ADDI,ADDIU,ANDI,XORI,ORI,SLTI,SLTIU,I_type;  
assign  ADDI=(op==ADDI_op);   
assign  ADDIU=(op==ADDIU_op);  
assign  ANDI=(op==ANDI_op);  
assign  XORI=(op==XORI_op);  
assign  SLTI=(op==SLTI_op);  
assign  SLTIU=(op==SLTIU_op);  
assign  ORI=(op==ORI_op);   
assign  I_type=ADDI||ADDIU||ANDI||XORI||ORI||SLTI||SLTIU; 


//SW,LW
localparam SW_op = 6'b101011;  
localparam LW_op = 6'b100011;  
wire SW,LW;  
assign SW=(op==SW_op);  
assign LW=(op==LW_op) ; 
 
 
//J
localparam J_op=6'b000010;  
assign J=(op==J_op); 


//JR
assign JR=(op==R_type_op)&&(funct==JR_funct);


//other control signal
assign RegWrite=LW||R_type1||R_type2||I_type; //需要写reg的    
assign RegDst=R_type1||R_type2;     //以rd为目的地址，否则是rt
assign MemWrite=SW;     
assign MemRead=LW;     
assign MemtoReg=LW;     
assign ALUSrcA=R_type2;     //以zero extend shamt为A
assign ALUSrcB=LW||SW||I_type; //sign edrend imm 为B
//J,JR,Branch 没有使用ALU
 

//ALUCode
localparam alu_add=  5'b00000;     //+ 
localparam alu_and=  5'b00001;     //&
localparam alu_xor=  5'b00010;     //^
localparam alu_or =  5'b00011;     //|
localparam alu_nor=  5'b00100;     //~^
localparam alu_sub=  5'b00101;     //-
localparam alu_andi= 5'b00110;     //-
localparam alu_xori= 5'b00111;  	//^
localparam alu_ori = 5'b01000;  	//|
localparam alu_jr =  5'b01001;  	//0
localparam alu_beq=  5'b01010;     //0
localparam alu_bne=  5'b01011;  	//0
localparam alu_bgez= 5'b01100;     	//0
localparam alu_bgtz= 5'b01101;    //0
localparam alu_blez= 5'b01110;     //0
localparam alu_bltz= 5'b01111; 		//0
localparam alu_sll=  5'b10000;  	//<<
localparam alu_srl=  5'b10001;  	//>>
localparam alu_sra=  5'b10010;   	//>>>
localparam alu_slt=  5'b10011;     //
localparam alu_sltu= 5'b10100;
localparam alu_addu= 5'b10101;
localparam alu_subu= 5'b10110;
 
always@(*)     
begin 
	if(op==R_type_op)        
	begin          
		case(funct)            
			ADD_funct  :ALUCode<=alu_add;            
			ADDU_funct :ALUCode<=alu_addu;            
			AND_funct  :ALUCode<=alu_and;            
			XOR_funct  :ALUCode<=alu_xor;            
			OR_funct   :ALUCode<=alu_or;            
			NOR_funct  :ALUCode<=alu_nor;            
			SUB_funct  :ALUCode<=alu_sub;            
			SUBU_funct :ALUCode<=alu_subu;
			SLT_funct  :ALUCode<=alu_slt;            
			SLTU_funct :ALUCode<=alu_sltu;            
			SLL_funct  :ALUCode<=alu_sll;           
			SLLV_funct :ALUCode<=alu_sll;            
			SRL_funct  :ALUCode<=alu_srl;            
			SRLV_funct :ALUCode<=alu_srl;            
			SRA_funct  :ALUCode<=alu_sra;            
			default    :ALUCode<=alu_sra;          
		endcase        
	end
	else        
	begin          
		case(op)            
		BEQ_op  :ALUCode<=alu_beq;            
		BNE_op  :ALUCode<=alu_bne;              
		BGEZ_op :begin 
					if(rt==BGEZ_rt) 
						ALUCode<=alu_bgez;
				end            
		BGTZ_op :begin 
					if(rt==BGTZ_rt) 
						ALUCode<=alu_bgtz;
				end            
		BLEZ_op :begin 
					if(rt==BLEZ_rt) 
						ALUCode<=alu_blez;
				end            
		BLTZ_op :begin 
					if(rt==BLTZ_rt) 
						ALUCode<=alu_bltz;
				end            
		ADDI_op :ALUCode<=alu_add;            
		ADDIU_op:ALUCode<=alu_addu;            
		ANDI_op :ALUCode<=alu_andi;            
		XORI_op :ALUCode<=alu_xori;            
		ORI_op  :ALUCode<=alu_ori;            
		SLTI_op :ALUCode<=alu_slt;            
		SLTIU_op:ALUCode<=alu_sltu;            
		SW_op   :ALUCode<=alu_add;            
		LW_op   :ALUCode<=alu_add;            
		default :ALUCode<=alu_add;          
		endcase        
	end     
end 
endmodule
