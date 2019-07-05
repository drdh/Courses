.ORIG x5000
Fx	;x in R0 return R4
	;ST R0,fSaveR0	;X
	ST R1,fSaveR1
	ST R2,fSaveR2
	ST R3,fSaveR3
	;ST R4,fSaveR4	;return
	ST R5,fSaveR5
	ST R6,fSaveR6
	;ST R7,fSaveR7	;PC
	

	LD R1,fInput	;R1=x4003
	LDR R2,R1,#0	;R2=degree
	ADD R1,R1,#1	;R1=x4004
	LDR R4,R1,#0	;R4=A
	;AND R4,R4,#0
	
fBLoop	ADD R2,R2,#-1
		BRn FxOver	
		AND R5,R5,#0
		ADD R3,R0,#0	;R3=x
		BRzp fSLoop		;R3>=0
		NOT R3,R3
		ADD R3,R3,#1
	fSLoop	ADD R3,R3,#-1
			BRn fBBr
			ADD R5,R5,R4
		BRnzp fSLoop
	fBBr	ADD R0,R0,#0
			BRzp NoM
			NOT R5,R5
			ADD R5,R5,#1
		NoM	AND R4,R4,#0
			ADD R4,R4,R5
			ADD R1,R1,#1
			LDR R6,R1,#0
			ADD R4,R4,R6
	BRnzp fBLoop

FxOver	
	;LD R0,fSaveR0	;X
	LD R1,fSaveR1
	LD R2,fSaveR2
	LD R3,fSaveR3
	;LD R4,fSaveR4	;return
	LD R5,fSaveR5
	LD R6,fSaveR6
	;LD R7,fSaveR7	;PC
	RET

;fSaveR0		.BLKW 1
fSaveR1		.BLKW 1
fSaveR2		.BLKW 1
fSaveR3		.BLKW 1
;SaveR4		.BLKW 1
fSaveR5		.BLKW 1
fSaveR6		.BLKW 1		
;fSaveR7		.BLKW 1
fInput		.FILL x4003
.END