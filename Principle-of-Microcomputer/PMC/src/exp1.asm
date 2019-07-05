	.model small
	.data
array	DB 36 DUP(0)
TEN		DB 10
SIX		DB 6		

	.code
	.startup
	LEA	BX,array
	mov CX,36
	mov DL,1
	
fill_array:
	mov [BX],DL
	inc DL
	inc BX
	loop fill_array
	
	LEA BX,array
	xor CX,CX ;AH:row,AL:col 
	
big_loop:
	cmp CH,CL	;row>=col
    JGE	print_num
	
print_return:
	CMP CL,5
    JE  print_CRLF

CRLF_return:	
	inc CL	;update row and col
	mov AL,CL
	mov AH,0
	div SIX
	mov CL,AH
	add CH,AL
	inc BX
	
	cmp CH,6;row==6
	JZ over
	jmp big_loop
	
print_num:
	mov	AL,[BX]	;AX to print
	mov	AH,0
	div	TEN	;AL=AX/10 AH=AX%10
	add AL,'0'
	add AH,'0'
	
	mov DX,AX	;print DL 
	mov AH,2	;select char print 
	INT 21H		;print tens
	
	SHR DX,8	;mov DL,DH
	INT 21H		;print ones
	
	mov DL,' '
	INT 21H		;print space
	jmp print_return
	
print_CRLF:
	mov AH,2
	mov DL,13
	INT 21H
	mov DL,10
	INT 21H
	jmp CRLF_return
	
over:
	.exit
	end
	
	
	
	