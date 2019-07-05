	.model small
	.data
	BIT = 20	;longest bits of answer
	ans db 21 DUP(0) ;store answe
	N	db 19
	TEN	db 10
	
	.code
	.startup
	
	mov DL,0	;current number
input:
	;input a char @AL
	mov AH,1
	INT 21H
	
	sub AL,'0'
	js finish_input ;not a digit
	
	push AX
	mov AL,DL
	;mov DL,10
	mul TEN		;AL=AL*10
	mov DL,AL
	pop AX
	add DL,AL	;DL=DL*10+AL
jmp input
	
finish_input:
	mov N,DL
	
	;output newline
	mov DL,10
	mov AH,2
	INT 21H
	mov Dl,13
	INT 21H
	
	;current number to mul CL
	mov CL,N
	mov ans,1

;begin to calculate
fact:
	cmp CX,1
	jz	print	;CX==1, calculate over

	mov BX ,0 ;ans[BX] to store result
	multi:
		cmp BX,BIT
		jz	carry_init
		mov AL,CL	;AL,number like 3
		mul BYTE PTR ans[BX];AL=AL*ans[BX]
		mov ans[BX],AL
		inc BX 
	jmp multi
	
	carry_init:
		mov BX,0
	
	carry:
		cmp BX,BIT
		jz	carry_over
		mov AX,0
		mov AL,ans[BX]
		div BYTE PTR TEN
		mov ans[BX],AH
		add ans[BX+1],AL
		inc BX
	jmp carry

	carry_over:
		dec CX
jmp fact
	

print:
	mov BX,BIT+1
	
find_non_zero:
	dec BX
	mov DL,ans[BX]
	cmp DL,0
	jz find_non_zero
	
print_digit:
	add DL,'0'
	mov AH,2
	INT 21H
	
	dec BX
	js over 
	mov DL,ans[BX]
jmp print_digit
	
over:
    .EXIT
    END