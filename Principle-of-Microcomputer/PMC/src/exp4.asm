	.model small
	.stack 3000H
	.data
error_msg 	db	0DH,0AH,"error!",0DH,0AH,'$'
CRLF		db 	0DH,0AH,'$'
expression	dw 	2048 DUP(0)
exp_pointer	dw  0
legal_char	db	"0123456789+-()$",0 ;end up with 0

last_exp	dw	0
is_last_char_num	db 	0	;0-not 

present_char	db	0
present_number	dw	0

division dw	10000,1000,100,10,1,'$'
result	db 6 DUP(0),'$'

jmp_return dw 0

correct_msg 	db	"done!",0DH,10,'$'

;1:Num,2:+,3:-,4:(,5:),6:$
	.code
	.startup
	LEA SI,expression
	mov [SI],'$'
	add SI,2	;2 bytes
	mov [SI],6	;6 present '$' as I defined
	add SI,2	;point to new position
	mov exp_pointer,SI
	
	;input
	call input
	cmp AH,0
	jz	main_print_error
	;calculate
	call calculate
	cmp AH,0
	jz	main_print_error
	
	;print result
	LEA SI,CRLF
	call print_string
	
	LEA SI,correct_msg
	call print_string
	
	LEA SI,result
	call print_string
	jmp over
	
	main_print_error:
		LEA	SI,error_msg
		call print_string
over:
	.exit
	
	
;from right to left	
;do not use any other subproc in this proc before finishing cal
;output:AH==1,correct
calculate proc
	;push expression into stack
	mov SI,last_exp
	push [SI-2];'$'
	push [SI];6,push as a end
	sub SI,4

	push_begin:	
		mov	DX,[SI]
		cmp DX,6
		jz encounter_dollar
		cmp DX,4
		jz encounter_left_parenthesis	;calculate the (...) first
		push [SI-2];only number can be pushed into stack
		push [SI]
		sub SI,4
	jmp push_begin
		
		encounter_dollar:
			mov jmp_return,encounter_dollar
			call cal_add_or_sub
			cmp AX,6
			jz	calculate_finish
			jmp calculate_wrong
			
		encounter_left_parenthesis:
			mov jmp_return,encounter_left_parenthesis
			call cal_add_or_sub
			cmp AX,5
			jz	encounter_right_parenthesis
			jmp calculate_wrong
			
			encounter_right_parenthesis:
				pop CX	;)
				push DX
				push 1
				sub	SI,4;discard 4,( 
				jmp push_begin
							
	
	calculate_finish:;DX result
		call output2result
		mov AH,1
		ret
	calculate_wrong:
		mov AH,0
		ret
calculate endp	

cal_add_or_sub proc
	pop BX
	
	pop AX
	cmp AX,1	;before the end dollar,must be a num
	jnz calculate_wrong
	pop DX	;num
	pop AX	;label
	cmp AX,2
	jz	add_
	cmp AX,3
	jz sub_
	
	push BX
	ret
	
	add_:
		pop CX	;+
		pop CX	;1
		pop	CX	;num
		add DX,CX
		push DX
		push 1
		jmp jmp_return
	sub_:
		pop CX	;-
		pop CX	;1
		pop	CX	;num
		sub DX,CX
		push DX
		push 1
		jmp jmp_return	
cal_add_or_sub endp

;input DX,the result
output2result proc
	LEA SI,result
	cmp DX,0		
	jns positive
		mov [SI],'-'
		inc SI
		neg	DX
	positive:
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
		divide_end:	ret
output2result endp


	
;input the expression, include check
;return 
;	AH==1,OK
;	AH==0,error
input proc 
input_begin:
	mov AH,1
	INT 21H
	mov present_char,AL
	
	;check present char
	call check_char ;AL:char
	cmp	AH,0
	jz	input_wrong
	
	;process the legal char
	mov AL,present_char
	call process_char
	cmp AH,0
	jz	input_begin
	
	input_right:
		mov AH,1
		ret
	input_wrong:
		mov AH,0
		ret
input endp

;process_char
;output AH=1,input over
;		AH=0,
process_char proc
	mov DL,present_char
	cmp	DL,'0'	; [ascii] +-()$ < 0~9
	js	process_char_not_number
		;present char a number
		sub	DL,'0'
		mov DH,0
		mov CX,DX
		mov is_last_char_num,1
		mov AX,present_number
		mov BX,10
		mul BX
		add AX,CX
		mov present_number,AX
		jmp process_char_ret
		
		;present char not a number
		process_char_not_number:
			mov AL,is_last_char_num
			cmp AL,0
			jz	process_char_not_number_last_not_num
				;last char is a number
				mov	AX,present_number;store present number
				mov SI,exp_pointer
				mov [SI],AX
				add SI,2
				mov [SI],1
				add SI,2
				mov exp_pointer,SI
				jmp not_unary
			process_char_not_number_last_not_num:
				mov DL,present_char
				cmp DL,'-'
				jz might_unary
				cmp DL,'+'
				jz might_unary
				jmp not_unary
				might_unary:
					;might be unary
					mov SI,exp_pointer	
					mov	BL,[SI-2]
					cmp	BL,4 ;(
					jz	unary
					cmp BL,6 ;$
					jz	unary
					jmp not_unary
					unary:
						mov word ptr[SI],0;insert a 0,(-2) --> (0-2)
						add SI,2
						mov word ptr[SI],1;1-num
						add SI,2
						mov exp_pointer,SI
						;add a '-+' like not a unary
				not_unary:;add non-number
					mov DL,present_char
					mov DH,0
					mov [SI],DL
					add SI,2
					
					cmp DL,'+'
					jz	plus
					cmp DL,'-'
					jz	minus
					cmp DL,'('
					jz left_parenthesis
					cmp DL,')'
					jz right_parenthesis
					cmp DL,'$'
					jz dollar
					
					plus:
						mov	DL,2
						jmp process_char_end
					minus:
						mov	DL,3
						jmp process_char_end
					left_parenthesis:
						mov	DL,4
						jmp process_char_end
					right_parenthesis:
						mov	DL,5
						jmp process_char_end
					dollar:
						mov DX,6
						mov [SI],DX
						mov last_exp,SI
						mov exp_pointer,SI
						mov AH,1
						ret
						
					process_char_end:		
						mov [SI],DX
						add SI,2
						mov AX,0
						mov is_last_char_num,AL
						mov present_number,AX
						mov exp_pointer,SI
	process_char_ret: 
		mov AH,0
		ret
process_char endp	



;check the current char
;input 	AL:char
;output	AH:1-right,0-wrong
check_char	proc
	LEA BX,legal_char
check_char_begin:
	mov DL,[BX]
	cmp DL,0
	jz 	check_char_wrong	;legal_char -->0 as a end
	cmp DL,AL
	jz	check_char_right ;just the present char correct
	inc	BX
	jmp check_char_begin
		
	check_char_wrong:
		mov AH,0
		ret
	check_char_right:
		mov AH,1
		ret
check_char 	endp

	
	
;print any string end up with '$'	
;[SI]as string source
;	LEA SI,error_msg
;	call print_string
print_string proc
	push DX
	push SI
	push AX
	
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