	.model small
	.stack 100H
	.data
buffer	db 0
file	db "2.txt",0 ;put the file in MyBuild
handle	dw 0
num		db 1024	DUP(0)
total	db 0
HUNDRED db 100
TEN 	db 10

	.code
	.startup
	mov AH,3Dh ;open file
	mov AL,0	;read-only
	LEA	DX,file	;file name in DX
	INT 21h
	jc over		;all error in c
	;open successful
	mov handle,AX	;store the handle
	LEA BX,num	;base of stored number
	mov CX,0	;count the total number
	
read_num:
	mov DX,0	;the number
add_digit:;0-255 like 123 the 1->2->3
	;store regs BX,CX
	push BX
	push CX
	push DX
	
	;read a char
	mov BX,handle
	mov AH,3Fh	;read char
	LEA DX,buffer
	mov CX,1	;only read one char
	INT 21h
	
	;restore
	pop DX
	pop CX
	pop BX
	
	;test EOF-0
	test AX,AX
	jz finish_read
	
	mov DH,buffer ;DH the char read
	;test whether DH a digit
	cmp	DH,'0'
    jb	store_num    
    cmp DH,'9'
    ja  store_num
	sub DH,'0'	;
	
	;mul DL
	;AX = AL * DL
	mov AL,10
	mul DL	
	mov DL,AL	;0-255
	add DL,DH	;DL=DL*10+DH
	jmp add_digit
	
store_num:	
	cmp buffer,' '
	jnz finish_read
	
	inc CX
	mov [BX],DL	;store number
	inc BX
	jmp read_num
	
finish_read:
	mov	total,CL
	
sort:
	dec CL
	jz finish_sort
	mov BL,1

bubble_sort:
	cmp BL,CL
	jz sort
	mov DL,num[BX-1]
	mov DH,0
	mov AL,NUM[BX]
	mov AH,0
	cmp AX,DX
	jns	skip_swap
	mov num[BX-1],AL
	mov NUM[BX],DL
skip_swap:
	inc BL
	jmp bubble_sort
	
finish_sort:
	mov BX,0

output:
	cmp BL,total
	jz over
	
	mov AL,NUM[BX]
	mov AH,0
	div HUNDRED
	mov DX,AX
	add DL,'0'
	mov AH,2
	INT 21H
	
	mov AH,0
	mov AL,DH
	div TEN
	mov DX,AX
	add Dl,'0'
	mov AH,2
	INT 21H
	
	mov DL,DH
	add Dl,'0'
	mov AH,2
	INT 21H
	
	mov dl,' '
	mov AH,2
	INT 21H
	
	inc BX
	jmp output
	
over:
    .EXIT
    end	