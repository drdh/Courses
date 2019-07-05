	.data
msg1: .asciiz "please enter 10 numbers:\n"
msg2: .asciiz "the sorted numbers are as follows:\n"
msg3: .asciiz "Num "
msg4: .asciiz " : "
	.text
main:
	addi $sp,$sp,-44
	sw $ra,0($sp)
	
	li $v0,4
	la $a0,msg1
	syscall
	
	li $t0,0
	li $t1,10
	move $t2,$sp
	
enter:
	beq $t0,$t1,enter_over
	addi $t2,$t2,4
	
	li $v0,4
	la $a0,msg3
	syscall
	
	addi $a0,$t0,1
	li $v0,1
	syscall
	
	li $v0,4
	la $a0,msg4
	syscall
	
	li $v0,5
	syscall
	
	sw $v0,0($t2)
	
	addi $t0,$t0,1
	j enter
enter_over:
	
	addi $a0,$sp,4
	li $a1,10
	
	jal sort

	li $t0,0
	li $t1,10
	move $t2,$sp
	
	li $v0,4
	la $a0,msg2
	syscall
	
print:
	beq $t0,$t1,print_over
	addi $t2,$t2,4
	lw $a0,0($t2)
	
	li $v0,1
	syscall
	
	li $a0,10
	li $v0,11
	syscall
	
	addi $t0,$t0,1
	j print
print_over:
	lw $ra,0($sp)
	addi $sp,$sp,44
jr $ra
	
	
sort:
	addi $sp,$sp,-20
	sw $ra,16($sp)
	sw $s3,12($sp)
	sw $s2,8($sp)
	sw $s1,4($sp)
	sw $s0,0($sp)
	
	move $s2,$a0
	move $s3,$a1
	
	move $s0,$zero
for1tst:
	slt $t0,$s0,$s3
	beq $t0,$zero,exit1
	
	addi $s1,$s0,-1
for2tst:
	slti $t0,$s1,0
	bne $t0,$zero,exit2
	sll $t1,$s1,2
	add $t2,$s2,$t1
	lw $t3,0($t2)
	lw $t4,4($t2)
	slt $t0,$t4,$t3
	beq $t0,$zero,exit2
	
	move $a0,$s2
	move $a1,$s1
	jal swap
	
	addi $s1,$s1,-1
	j for2tst

exit2:
	addi $s0,$s0,1
	j for1tst
	
exit1:
	lw $s0,0($sp)
	lw $s1,4($sp)
	lw $s2,8($sp)
	lw $s3,12($sp)
	lw $ra,16($sp)
	addi $sp,$sp,20
	
	jr $ra
	
	
swap:
	sll $t1,$a1,2
	add $t1,$a0,$t1
	
	lw $t0,0($t1)
	lw $t2,4($t1)
	
	sw $t2,0($t1)
	sw $t0,4($t1)
	
	jr $ra