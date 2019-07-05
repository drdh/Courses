.ORIG x3000
	LD R7,Left
	LDR R1,R7,#0
	ST R1,LowX
	LD R7,Right
	LDR R5,R7,#0
	
	ST R5,HighX
	
	ADD R0,R1,#0
	ST R7,SaveR7
	LD R7,Fx
	JSRR R7
	LD R7,SaveR7
	ADD R2,R4,#0
	ST R2,LowF
	
	ADD R0,R5,#0
	ST R7,SaveR7
	LD R7,Fx
	JSRR R7
	LD R7,SaveR7
	ADD R6,R4,#0
	ST R6,HighF
	
	AND R7,R7,#0
	
	NOT R2,R2
	ADD R2,R2,#1
	ADD R2,R2,R6
	BRnz LgtH
	ADD R7,R7,#1
LgtH	ST R7,Monotonic
	
	ST R7,SaveR7
	JSR Divide
	LD R7,SaveR7
	
	ADD R0,R4,#0
	ST R4,MiddleX
	
	ST R7,SaveR7
	LD R7,Fx
	JSRR R7
	LD R7,SaveR7
	
	ST R4,MiddleF
	
MLoop	LD R4,MiddleF
		BRz MainOver
		BRn BElse
			LD R7,Monotonic
			BRnz SElse1
			LD R1,MiddleX
			ST R1,HighX
			BRnzp Compu
			
			SElse1
			LD R1,MiddleX
			ST R1,LowX
			
		BRnzp Compu
		BElse
			LD R7,Monotonic
			BRnz SElse2
			LD R1,MiddleX
			ST R1,LowX
			BRnzp Compu
			SElse2
			LD R1,MiddleX
			ST R1,HighX
		
	Compu	
	ST R7,SaveR7
	JSR Divide
	LD R7,SaveR7
	ST R4,MiddleX
	LD R0,MiddleX
	ST R7,SaveR7
	LD R7,Fx
	JSRR R7
	ST R4,MiddleF
	BRnzp MLoop
	

MainOver
	LD R4,MiddleX
	LD R7,SaveZero
	STR R4,R7,#0
HALT



Divide	;use LowX HighX,return R4
		;ST R0,SaveR0	;X
		ST R1,SaveR1
		ST R2,SaveR2
		ST R3,SaveR3
		;ST R4,SaveR4	;return
		ST R5,SaveR5
		ST R6,SaveR6
		;ST R7,SaveR7	;PC
		
		
		LD R1,LowX
		LD R2,HighX
		ADD R0,R1,R2
		BRn	DiBr1
		AND R4,R4,#0
		BRnzp DiSLoop
DiBr1	;R0<0
		AND R4,R4,#0
		ADD R4,R4,x0003

DiSLoop	AND R1,R1,#0
		ADD R1,R1,#14	;R1=14
DiLoop	
		ADD R1,R1,#-1
		BRn DiOver
		ADD R4,R4,R4	;<<
		ADD R0,R0,R0	;<<
		BRzp DiLoop
		ADD R4,R4,#1
		BRnzp DiLoop
		
DiOver	;LD R0,SaveR0	;X
		LD R1,SaveR1
		LD R2,SaveR2
		LD R3,SaveR3
		;LD R4,SaveR4	;return
		LD R5,SaveR5
		LD R6,SaveR6
		RET


LowX		.BLKW 1
HighX		.BLKW 1
LowF		.BLKW 1
HighF		.BLKW 1
MiddleX		.BLKW 1
MiddleF		.BLKW 1
Monotonic	.BLKW 1
Fx			.FILL x5000
;SaveR0		.BLKW 1
SaveR1		.BLKW 1
SaveR2		.BLKW 1
SaveR3		.BLKW 1
;SaveR4		.BLKW 1
SaveR5		.BLKW 1
SaveR6		.BLKW 1
SaveR7		.BLKW 1
Left		.FILL x4001
Right		.FILL x4002
SaveZero	.FILL x4000
.END









