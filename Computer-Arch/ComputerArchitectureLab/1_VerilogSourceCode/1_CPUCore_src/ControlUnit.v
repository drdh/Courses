`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: USTC ESLAB (Embeded System Lab)
// Engineer: Haojun Xia
// Create Date: 2019/02/08
// Design Name: RISCV-Pipline CPU
// Module Name: ControlUnit
// Target Devices: Nexys4
// Tool Versions: Vivado 2017.4.1
// Description: RISC-V Instruction Decoder
//////////////////////////////////////////////////////////////////////////////////
`include "Parameters.v"   
module ControlUnit(
    input wire [6:0] Op,
    input wire [2:0] Fn3,
    input wire [6:0] Fn7,
    output wire JalD,
    output wire JalrD,
    output reg [2:0] RegWriteD,
    output wire MemToRegD,
    output reg [3:0] MemWriteD,
    output wire LoadNpcD,
    output wire [1:0] RegReadD,//原来是reg
    output reg [2:0] BranchTypeD,
    output reg [3:0] AluContrlD,
    output wire [1:0] AluSrc2D,
    output wire AluSrc1D,
    output reg [2:0] ImmType        
    );

    //先定义几个常用的用于比较的常量
    localparam lui_op=  7'b0110111;
    localparam auipc_op=7'b0010111;
    localparam jal_op=  7'b1101111;
    localparam jalr_op= 7'b1100111;
    localparam br_op=   7'b1100011;
    localparam load_op= 7'b0000011;
    localparam store_op=7'b0100011;
    localparam IType_op=7'b0010011; //带I的指令，不全是I-Type
    localparam RType_op=7'b0110011; //类似上面

    //用于判断具体指令，当且仅当为该指令时，为1'b1
    wire LUI,AUIPC,JAL,JALR;
    wire BEQ,BNE,BLT,BGE,BLTU,BGEU;
    wire LB,LH,LW,LBU,LHU;
    wire SB,SH,SW;
    wire ADDI,SLTI,SLTIU,XORI,ORI,ANDI,SLLI,SRLI,SRAI;
    wire ADD,SUB,SLL,SLT,SLTU,XOR,SRL,SRA,OR,AND;

    //由Op唯一指定
    assign LUI= (Op == lui_op);
    assign AUIPC= (Op == auipc_op);
    assign JAL = (Op == jal_op);
    assign JALR = (Op == jalr_op);

    //br
    assign BEQ = (Op == br_op)&&(Fn3 == 3'b000);
    assign BNE = (Op == br_op)&&(Fn3 == 3'b001);
    assign BLT = (Op == br_op)&&(Fn3 == 3'b100);
    assign BGE = (Op == br_op)&&(Fn3 == 3'b101);
    assign BLTU = (Op == br_op)&&(Fn3 == 3'b110);
    assign BGEU = (Op == br_op)&&(Fn3 == 3'b111);

    //load
    assign LB = (Op == load_op)&&(Fn3 == 3'b000);
    assign LH = (Op == load_op)&&(Fn3 == 3'b001);
    assign LW = (Op == load_op)&&(Fn3 == 3'b010);
    assign LBU = (Op == load_op)&&(Fn3 == 3'b100);
    assign LHU = (Op == load_op)&&(Fn3 == 3'b101);

    //store
    assign SB = (Op == store_op)&&(Fn3 == 3'b000);
    assign SH = (Op == store_op)&&(Fn3 == 3'b001);
    assign SW = (Op == store_op)&&(Fn3 == 3'b010);

    //algo I
    assign ADDI = (Op == IType_op )&&(Fn3 == 3'b000);
    assign SLTI = (Op == IType_op )&&(Fn3 == 3'b010);
    assign SLTIU = (Op == IType_op )&&(Fn3 == 3'b011);
    assign XORI = (Op == IType_op )&&(Fn3 == 3'b100);
    assign ORI = (Op == IType_op )&&(Fn3 == 3'b110);
    assign ANDI = (Op == IType_op )&&(Fn3 == 3'b111);

    localparam Fn7_0=7'b0000000;
    localparam Fn7_1=7'b0100000;

    //shift I
    assign SLLI= (Op == IType_op)&&(Fn3 == 3'b001)&&(Fn7 == Fn7_0);
    assign SRLI= (Op == IType_op)&&(Fn3 == 3'b101)&&(Fn7 == Fn7_0);
    assign SRAI= (Op == IType_op)&&(Fn3 == 3'b101)&&(Fn7 == Fn7_1);

    //R
    assign ADD = (Op == RType_op)&&(Fn3 == 3'b000)&&(Fn7 == Fn7_0);
    assign SUB = (Op == RType_op)&&(Fn3 == 3'b000)&&(Fn7 == Fn7_1);
    assign SLL = (Op == RType_op)&&(Fn3 == 3'b001)&&(Fn7 == Fn7_0);
    assign SLT = (Op == RType_op)&&(Fn3 == 3'b010)&&(Fn7 == Fn7_0);
    assign SLTU = (Op == RType_op)&&(Fn3 == 3'b011)&&(Fn7 == Fn7_0);
    assign XOR = (Op == RType_op)&&(Fn3 == 3'b100)&&(Fn7 == Fn7_0);
    assign SRL = (Op == RType_op)&&(Fn3 == 3'b101)&&(Fn7 == Fn7_0);
    assign SRA = (Op == RType_op)&&(Fn3 == 3'b101)&&(Fn7 == Fn7_1);
    assign OR = (Op == RType_op)&&(Fn3 == 3'b110)&&(Fn7 == Fn7_0);
    assign AND = (Op == RType_op)&&(Fn3 == 3'b111)&&(Fn7 == Fn7_0);

    //处理输出
    assign JalD=JAL;
    assign JalrD=JALR;

/*
//RegWrite[2:0]  six kind of ways to save values to Register
    `define NOREGWRITE  3'b0	//	Do not write Register
    `define LB  3'd1			//	load 8bit from Mem then signed extended to 32bit
    `define LH  3'd2			//	load 16bit from Mem then signed extended to 32bit
    `define LW  3'd3			//	write 32bit to Register
    `define LBU  3'd4			//	load 8bit from Mem then unsigned extended to 32bit
    `define LHU  3'd5			//	load 16bit from Mem then unsigned extended to 32bit
*/
    //output reg [2:0] RegWriteD,表示ID阶段的指令对应的 寄存器写入模式
    //作用在Data Ext上，是对mem的数据进行处理,访存的有LB,LH,LW,LBU,LHU

    //其他要写入reg的
    wire OtherWriteReg;//除了load的几个
    assign OtherWriteReg=(LUI||AUIPC||JAL||JALR)||(Op==IType_op)||(Op==RType_op);

    always@(*)
    begin
        case ({LB,LH,LW,LBU,LHU,OtherWriteReg})
            6'b100000: RegWriteD <= `LB;
            6'b010000: RegWriteD <= `LH;
            6'b001000: RegWriteD <= `LW;
            6'b000100: RegWriteD <= `LBU;
            6'b000010: RegWriteD <= `LHU;
            6'b000001: RegWriteD <= `LW;
            default: RegWriteD <= `NOREGWRITE;
        endcase
    end

    //output wire MemToRegD,
    // MemToRegD==1     表示ID阶段的指令需要将data memory读取的值写入寄存器    
    assign MemToRegD=(LB || LH || LW || LBU || LHU);

    //output reg [3:0] MemWriteD,采用独热码格式，对于data memory的32bit字按byte进行写入,MemWriteD=0001表示只写入最低1个byte
    //仅仅有SB,SH,SW有存储mem功能
    always@(*)
    begin
        case ({SB,SH,SW})
            3'b100: MemWriteD <= 4'b0001;
            3'b010: MemWriteD <= 4'b0011;
            3'b001: MemWriteD <= 4'b1111;
            default: MemWriteD <= 4'b0000;
        endcase
    end

    //output wire LoadNpcD,表示将NextPC输出到ResultM
    //需要nextPC写入到RD的JAL,JALR
    assign LoadNpcD=(JAL||JALR);

    //output reg [1:0] RegReadD,已改成wire类型
    // RegReadD[1]==1   表示A1对应的寄存器值被使用到了，RegReadD[0]==1表示A2对应的寄存器值被使用到了，用于forward的处理
    assign RegReadD[1]=JALR||(Op==br_op)||(Op==load_op)||(Op==store_op)||(Op==IType_op)||(Op==RType_op);
    assign RegReadD[0]=(Op==br_op)||(Op==store_op)||(Op==RType_op);

    //output reg [2:0] BranchTypeD,表示不同的分支类型
/*
//BranchType[2:0]
    `define NOBRANCH  3'd0
    `define BEQ  3'd1
    `define BNE  3'd2
    `define BLT  3'd3
    `define BLTU  3'd4
    `define BGE  3'd5
    `define BGEU  3'd6
*/
    always@(*)
    begin
        case ({BEQ,BNE,BLT,BLTU,BGE,BGEU})
            6'b100000: BranchTypeD <= `BEQ;
            6'b010000: BranchTypeD <= `BNE;
            6'b001000: BranchTypeD <= `BLT;
            6'b000100: BranchTypeD <= `BLTU;
            6'b000010: BranchTypeD <= `BGE;
            6'b000001: BranchTypeD <= `BGEU;
            default: BranchTypeD <= `NOBRANCH;
        endcase
    end

    //output reg [3:0] AluContrlD,表示不同的ALU计算功能
/*
/ALUContrl[3:0]
    `define SLL  4'd0
    `define SRL  4'd1
    `define SRA  4'd2
    `define ADD  4'd3
    `define SUB  4'd4
    `define XOR  4'd5
    `define OR  4'd6
    `define AND  4'd7
    `define SLT  4'd8
    `define SLTU  4'd9
    `define LUI  4'd10
*/  
    always@(*)
    begin
        if((Op==load_op)||(Op==store_op)||ADD||ADDI||AUIPC||JALR)begin AluContrlD <= `ADD; end
        else if(SUB)begin AluContrlD <= `SUB; end
        else if(LUI)begin AluContrlD <= `LUI; end 
        else if(XOR||XORI)begin AluContrlD <= `XOR; end
        else if(OR||ORI)begin AluContrlD <= `OR;  end
        else if(AND||ANDI)begin AluContrlD <= `AND; end
        else if(SLL||SLLI)begin AluContrlD <= `SLL; end
        else if(SRL||SRLI)begin AluContrlD <= `SRL; end
        else if(SRA||SRAI)begin AluContrlD <= `SRA; end
        else if(SLT||SLTI)begin AluContrlD <= `SLT; end
        else if(SLTU||SLTIU)begin AluContrlD <= `SLTU; end
        else begin AluContrlD <= 4'dx; end
    end
    
    //output wire [1:0] AluSrc2D,表示Alu输入源2的选择; 00:Reg   01:Rs2 5bits    10:Imm    
    //output wire AluSrc1D,表示Alu输入源1的选择; 0:Reg  1:PC 
    assign AluSrc1D = (AUIPC);
    assign AluSrc2D = (SLLI||SRLI||SRAI)? 2'b01 : ((Op==br_op||Op==RType_op)? 2'b00 : 2'b10);

    //output reg [2:0] ImmType;表示指令的立即数格式
/*
//ImmType[2:0]
    `define RTYPE  3'd0
    `define ITYPE  3'd1
    `define STYPE  3'd2
    `define BTYPE  3'd3
    `define UTYPE  3'd4
    `define JTYPE  3'd5
*/
    always@(*)
    begin
        if(Op==RType_op)begin ImmType<=`RTYPE; end
        else if(Op==IType_op || Op==load_op || JALR) begin ImmType<=`ITYPE; end
        else if(Op==store_op)begin ImmType<=`STYPE; end
        else if(Op==br_op)begin ImmType<=`BTYPE; end
        else if(LUI||AUIPC)begin ImmType<=`UTYPE; end
        else if(JAL)begin ImmType<=`JTYPE; end
        else begin ImmType<=3'dx; end
    end

endmodule

//功能说明
    //ControlUnit       是本CPU的指令译码器，组合逻辑电路
//输入
    // Op               是指令的操作码部分
    // Fn3              是指令的func3部分
    // Fn7              是指令的func7部分
//输出
    // JalD==1          表示Jal指令到达ID译码阶段
    // JalrD==1         表示Jalr指令到达ID译码阶段
    // RegWriteD        表示ID阶段的指令对应的 寄存器写入模式 ，所有模式定义在Parameters.v中
    // MemToRegD==1     表示ID阶段的指令需要将data memory读取的值写入寄存器,
    // MemWriteD        共4bit，采用独热码格式，对于data memory的32bit字按byte进行写入,MemWriteD=0001表示只写入最低1个byte，和xilinx bram的接口类似
    // LoadNpcD==1      表示将NextPC输出到ResultM
    // RegReadD[1]==1   表示A1对应的寄存器值被使用到了，RegReadD[0]==1表示A2对应的寄存器值被使用到了，用于forward的处理
    // BranchTypeD      表示不同的分支类型，所有类型定义在Parameters.v中
    // AluContrlD       表示不同的ALU计算功能，所有类型定义在Parameters.v中
    // AluSrc2D         表示Alu输入源2的选择
    // AluSrc1D         表示Alu输入源1的选择
    // ImmType          表示指令的立即数格式，所有类型定义在Parameters.v中   
//实验要求  
    //实现ControlUnit模块   