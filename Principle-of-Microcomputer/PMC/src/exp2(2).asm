	.model small
	.stack 100H
	.data
buffer	db 0
file	db "2.txt",0 ;put the file in MyBuild
handle	dw 0
num		dw 1024	DUP(0)
total	dw 0 ;addr 2*number
HUNDRED db 100
TEN 	dw 10
division dw	10000,1000,100,10,1,'$'
result	db 6 DUP(0),'$'


	.code
	.startup
	call read_file
	call sort 
	
output:
	cmp BX,CX
	ja  over
	mov DX,WORD PTR num[BX]
	call output2result
	LEA SI,result
	call print_string
	add BX,2
	jmp output
	
over:
    .EXIT

;name:file
;to:num(DW)	
;total num*2
read_file proc
	push BP
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
		
		mov AL,buffer ;AL the char read
		;test whether AL a digit
		cmp	AL,'0'
		jb	store_num    
		cmp AL,'9'
		ja  store_num
		sub AL,'0'	;AL current digit
		
		;mul TEN
		;(DX AX) = AX * 10
		mov AH,0
		push BX
		mov BX,AX
		mov AX,DX
		mul TEN	
		add BX,AX
		mov DX,BX
		
		pop BX
		jmp add_digit
		
	store_num:	
		cmp buffer,' '
		jnz finish_read
		
		inc CX
		mov WORD PTR[BX],DX	;store number
		add BX,2
		jmp read_num
		
	finish_read:
		add CX,CX
		mov	total,CX
		pop BP
		ret 
read_file endp	
	
	
;CX total	
sort proc
	mov CX,total
	sort_begin:	
		sub CX,2
		jz finish_sort
		mov BX,2

	bubble_sort:
		cmp BX,CX
		ja sort_begin
		mov DX,WORD PTR num[BX-2]
		mov AX,WORD PTR NUM[BX]
		cmp AX,DX
		jns	skip_swap
		mov WORD PTR num[BX-2],AX
		mov WORD PTR NUM[BX],DX
	skip_swap:
		add BX,2
		jmp bubble_sort
		
	finish_sort:
		mov CX,total
		mov BX,0	
		ret
sort endp
	
;input DX,the result
output2result proc
	push CX
	push BX
	LEA SI,result
	cmp DX,0		
;	jns positive
;		mov [SI],'-'
;		inc SI
;		neg	DX
;	positive:
	;AX=(DX AX)/BX
	;DX=(DX AX)%BX
	LEA BX,division
	divide_begin:
		mov CX,[BX]
		add BX,2
		cmp CX,'$'
		jz divide_end
		
		mov AX,DX
		mov DX,0
		div CX	
		add AL,'0'
		mov [SI],AL
		inc SI
	jmp divide_begin
	divide_end:	
	pop BX
	pop CX
	ret
output2result endp

;print any string end up with '$'	
;[SI]as string source
;	LEA SI,error_msg
;	call print_string
print_string proc
	push DX
	push SI
	push AX
	
	dec SI
	find_non_zero:
		inc SI
		mov DL,[SI]
		cmp DL,'0'
		jz find_non_zero
		
	string_loop:
		mov DL,[SI]
		cmp DL,'$'
		jz string_ret	
		mov AH,2
		INT 21H
		inc SI
	jmp string_loop
	string_ret:
	pop AX
	pop SI
	pop DX
	ret
print_string endp	
	
    end	