



参考4_ProjectDesignFiles 文件夹中提供的**CPU模块图**和1_VerilogSourceCode文件夹中对应的**Verilog代码**，思考每条指令的数据通路，详细写出**每个待完成模块**的设计思路，并思考如何用verilog简洁高效的表达这些逻辑电路。

RISC-V 32bit 整型指令集（除去FENCE,FENCE.I,CSR,ECALL和EBREAK指令）

请在报告中回答下述问题：

1. 为什么将DataMemory和InstructionMemory嵌入在段寄存器中？

2. DataMemory和InstructionMemory输入地址是字（32bit）地址，如何将访存地址转化为字地址输入进去？

3. 如何实现DataMemory的非字对齐的Load？

4. 如何实现DataMemory的非字对齐的Store？

5. 为什么RegFile的时钟要取反？

6. NPC_Generator中对于不同跳转target的选择有没有优先级？

7. ALU模块中，默认wire变量是有符号数还是无符号数？

8. AluSrc1E执行哪些指令时等于1’b1？

9. AluSrc2E执行哪些指令时等于2‘b01？

10. 哪条指令执行过程中会使得LoadNpcD==1？

11. DataExt模块中，LoadedBytesSelect的意义是什么？

12. Harzard模块中，有哪几类冲突需要插入气泡？

13. Harzard模块中采用默认不跳转的策略，遇到branch指令时，如何控制flush和stall信号？

14. Harzard模块中，RegReadE信号有什么用？

15. 0号寄存器值始终为0，是否会对forward的处理产生影响？



