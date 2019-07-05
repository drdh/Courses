.data
fibs: .word 0:20
size: .word 20
temp: .word 3 3
	.text
	la $t0,fibs
	la $t5,size
	lw $t5,0($t5)
	la $t3,temp
	lw $t3,0($t3)
	la $t4,temp
	lw $t4,4($t4)
	sw $t3,0($t0)
	sw $t4,4($t0)
	addi $t1,$t5,-2
loop:	lw $t3,0($t0)
	lw $t3,0($t0)
	lw $t4,4($t0)
	addi $t1,$t1,-1
	addi $t0,$t0,4
	add $t2,$t3,$t4	
	sw $t2,4($t0)	
	bgtz $t1,loop
out:	j out
