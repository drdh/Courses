.text
add $1,$0,$0
addi $1,$1,1
addiu $1,$1,1
addu $2,$1,$1
sub $3,$1,$2
subu $4,$1,$2
and $5,$1,$2
or $6,$1,$2
nor $7,$1,$2
xor $8,$1,$2
andi $9,$1,2
ori $10,$1,1
xori $11,$2,1
slt $12,$3,$t1
sltu $13,$3,$t1
addi $14,$0,-100
sll $15,$1,2
srl $16,$14,2
sra $17,$14,2
sllv $18,$1,$1
srlv $19,$14,$1
srav $20,$14,$1
slti $21,$1,-1
sltiu $22,$1,-1
beq $1,$2,beq_d
addi $23,$0,1
beq_d: bne $1,$2,bne_d
addi $24,$0,1
bne_d: bgez $0,bgez_d
addi $25,$0,1
bgez_d: bgtz $0,bgtz_d
addi $26,$0,1
bgtz_d:blez $0,blez_d
addi $27,$0,1
blez_d: bltz $0,bltz_d
addi $28,$0,1
bltz_d:la $29,out
jr $29
addi $30,$0,1
out: j out
