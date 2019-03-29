`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: USTC ESLAB（Embeded System Lab）
// Engineer: Haojun Xia & Xuan Wang
// Create Date: 2019/02/22
// Design Name: RISCV-Pipline CPU
// Module Name: HarzardUnit
// Target Devices: Nexys4
// Tool Versions: Vivado 2017.4.1
// Description: Deal with harzards in pipline
//////////////////////////////////////////////////////////////////////////////////
module HarzardUnit(
    input wire CpuRst, ICacheMiss, DCacheMiss, 
    input wire BranchE, JalrE, JalD, 
    input wire [4:0] Rs1D, Rs2D, Rs1E, Rs2E, RdE, RdM, RdW,
    input wire [1:0] RegReadE,
    input wire [2:0] MemToRegE, RegWriteM, RegWriteW,
    output reg StallF, FlushF, StallD, FlushD, StallE, FlushE, StallM, FlushM, StallW, FlushW,
    output wire [1:0] Forward1E, Forward2E //之前是reg
    );
    //Stall and Flush signals generate
    always@(*)
    begin
        if(CpuRst)begin
            StallF=1'b0;
            StallD=1'b0;
            StallE=1'b0;
            StallM=1'b0;
            StallW=1'b0;

            FlushF=1'b1;
            FlushD=1'b1;
            FlushE=1'b1;
            FlushM=1'b1;
            FlushW=1'b1;
        end
        else begin
            //LW
            if(MemToRegE && ((RdE==Rs1D )||(RdE==Rs2D))&& RdE!=0)begin
                StallF=1'b1; //F,D等待一个stall
                StallD=1'b1;
                StallE=1'b0;
                StallM=1'b0;
                StallW=1'b0;

                FlushF=1'b0;
                FlushD=1'b0;
                FlushE=1'b0;
                FlushM=1'b0;
                FlushW=1'b0;
            end
            else if(BranchE || JalrE)begin
                StallF=1'b0;
                StallD=1'b0;
                StallE=1'b0;
                StallM=1'b0;
                StallW=1'b0;

                FlushF=1'b0;  //猜测执行PC+4的指令失败
                FlushD=1'b1;
                FlushE=1'b1;
                FlushM=1'b0;
                FlushW=1'b0;
            end
            else if(JalD)begin
                StallF=1'b0;
                StallD=1'b0;
                StallE=1'b0;
                StallM=1'b0;
                StallW=1'b0;

                FlushF=1'b0;    //仅仅只用IF
                FlushD=1'b1;
                FlushE=1'b0;
                FlushM=1'b0;
                FlushW=1'b0;
            end
            else begin
                StallF=1'b0;
                StallD=1'b0;
                StallE=1'b0;
                StallM=1'b0;
                StallW=1'b0;

                FlushF=1'b0;
                FlushD=1'b0;
                FlushE=1'b0;
                FlushM=1'b0;
                FlushW=1'b0;
            end
        end
    end


    //Forward Register Source 1
    assign Forward1E[0]=(|RegWriteW)&&(RdW!=0)&&(!((RdM==Rs1E)&&(|RegWriteM)))&&(RdW==Rs1E)&&RegReadE[1];
    assign Forward1E[1]=(|RegWriteM)&&(RdM!=0)&&(RdM==Rs1E)&&RegReadE[1];

    //Forward Register Source 2
    assign Forward2E[0]=(|RegWriteW)&&(RdW!=0)&&(!((RdM==Rs2E)&&(|RegWriteM)))&&(RdW==Rs2E)&&RegReadE[0];
    assign Forward2E[1]=(|RegWriteM)&&(RdM!=0)&&(RdM==Rs2E)&&RegReadE[1];

endmodule

//功能说明
    //HarzardUnit用来处理流水线冲突，通过插入气泡，forward以及冲刷流水段解决数据相关和控制相关，组合逻辑电路
    //可以最后实现。前期测试CPU正确性时，可以在每两条指令间插入四条空指令，然后直接把本模块输出定为，不forward，不stall，不flush 
//输入
    //CpuRst                                    外部信号，用来初始化CPU，当CpuRst==1时CPU全局复位清零（所有段寄存器flush），Cpu_Rst==0时cpu开始执行指令
    //ICacheMiss, DCacheMiss                    为后续实验预留信号，暂时可以无视，用来处理cache miss
    //BranchE, JalrE, JalD                      用来处理控制相关
    //Rs1D, Rs2D, Rs1E, Rs2E, RdE, RdM, RdW     用来处理数据相关，分别表示源寄存器1号码，源寄存器2号码，目标寄存器号码
    //RegReadE RegReadD[1]==1                   表示A1对应的寄存器值被使用到了，RegReadD[0]==1表示A2对应的寄存器值被使用到了，用于forward的处理
    //RegWriteM, RegWriteW                      用来处理数据相关，RegWrite!=3'b0说明对目标寄存器有写入操作
    //MemToRegE                                 表示Ex段当前指令 从Data Memory中加载数据到寄存器中
//输出
    //StallF, FlushF, StallD, FlushD, StallE, FlushE, StallM, FlushM, StallW, FlushW    控制五个段寄存器进行stall（维持状态不变）和flush（清零）
    //Forward1E, Forward2E                                                              控制forward
//实验要求  
    //实现HarzardUnit模块   