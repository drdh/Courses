	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.globl	_NewTreeNode            ## -- Begin function NewTreeNode
	.p2align	4, 0x90
_NewTreeNode:                           ## @NewTreeNode
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi0:
	.cfi_def_cfa_offset 16
Lcfi1:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi2:
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movl	$16, %eax
	movl	%eax, %ecx
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rcx, %rdi
	callq	_malloc
	movq	%rax, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	-24(%rbp), %rcx
	movq	%rax, (%rcx)
	movq	-16(%rbp), %rax
	movq	-24(%rbp), %rcx
	movq	%rax, 8(%rcx)
	movq	-24(%rbp), %rax
	addq	$32, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	_ItemCheck              ## -- Begin function ItemCheck
	.p2align	4, 0x90
_ItemCheck:                             ## @ItemCheck
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi3:
	.cfi_def_cfa_offset 16
Lcfi4:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi5:
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -16(%rbp)
	movq	-16(%rbp), %rdi
	cmpq	$0, (%rdi)
	jne	LBB1_2
## BB#1:
	movq	$1, -8(%rbp)
	jmp	LBB1_3
LBB1_2:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdi
	callq	_ItemCheck
	addq	$1, %rax
	movq	-16(%rbp), %rdi
	movq	8(%rdi), %rdi
	movq	%rax, -24(%rbp)         ## 8-byte Spill
	callq	_ItemCheck
	movq	-24(%rbp), %rdi         ## 8-byte Reload
	addq	%rax, %rdi
	movq	%rdi, -8(%rbp)
LBB1_3:
	movq	-8(%rbp), %rax
	addq	$32, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	_BottomUpTree           ## -- Begin function BottomUpTree
	.p2align	4, 0x90
_BottomUpTree:                          ## @BottomUpTree
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi6:
	.cfi_def_cfa_offset 16
Lcfi7:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi8:
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movl	%edi, -12(%rbp)
	cmpl	$0, -12(%rbp)
	jbe	LBB2_2
## BB#1:
	movl	-12(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edi
	callq	_BottomUpTree
	movl	-12(%rbp), %edi
	subl	$1, %edi
	movq	%rax, -24(%rbp)         ## 8-byte Spill
	callq	_BottomUpTree
	movq	-24(%rbp), %rdi         ## 8-byte Reload
	movq	%rax, %rsi
	callq	_NewTreeNode
	movq	%rax, -8(%rbp)
	jmp	LBB2_3
LBB2_2:
	xorl	%eax, %eax
	movl	%eax, %ecx
	movq	%rcx, %rdi
	movq	%rcx, %rsi
	callq	_NewTreeNode
	movq	%rax, -8(%rbp)
LBB2_3:
	movq	-8(%rbp), %rax
	addq	$32, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	_DeleteTree             ## -- Begin function DeleteTree
	.p2align	4, 0x90
_DeleteTree:                            ## @DeleteTree
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi9:
	.cfi_def_cfa_offset 16
Lcfi10:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi11:
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	cmpq	$0, (%rdi)
	je	LBB3_2
## BB#1:
	movq	-8(%rbp), %rax
	movq	(%rax), %rdi
	callq	_DeleteTree
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	callq	_DeleteTree
LBB3_2:
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	callq	_free
	addq	$16, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3               ## -- Begin function main
LCPI4_0:
	.quad	4611686018427387904     ## double 2
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi12:
	.cfi_def_cfa_offset 16
Lcfi13:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi14:
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rsi
	movq	8(%rsi), %rdi
	callq	_atol
	movl	%eax, %ecx
	movl	%ecx, -20(%rbp)
	movl	$4, -28(%rbp)
	movl	-28(%rbp), %ecx
	addl	$2, %ecx
	cmpl	-20(%rbp), %ecx
	jbe	LBB4_2
## BB#1:
	movl	-28(%rbp), %eax
	addl	$2, %eax
	movl	%eax, -32(%rbp)
	jmp	LBB4_3
LBB4_2:
	movl	-20(%rbp), %eax
	movl	%eax, -32(%rbp)
LBB4_3:
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %edi
	callq	_BottomUpTree
	movq	%rax, -48(%rbp)
	movl	-36(%rbp), %esi
	movq	-48(%rbp), %rdi
	movl	%esi, -92(%rbp)         ## 4-byte Spill
	callq	_ItemCheck
	leaq	L_.str(%rip), %rdi
	movl	-92(%rbp), %esi         ## 4-byte Reload
	movq	%rax, %rdx
	movb	$0, %al
	callq	_printf
	movq	-48(%rbp), %rdi
	movl	%eax, -96(%rbp)         ## 4-byte Spill
	callq	_DeleteTree
	movl	-32(%rbp), %edi
	callq	_BottomUpTree
	movq	%rax, -56(%rbp)
	movl	-28(%rbp), %esi
	movl	%esi, -24(%rbp)
LBB4_4:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB4_6 Depth 2
	movl	-24(%rbp), %eax
	cmpl	-32(%rbp), %eax
	ja	LBB4_11
## BB#5:                                ##   in Loop: Header=BB4_4 Depth=1
	movl	-32(%rbp), %eax
	movl	-24(%rbp), %ecx
	subl	%ecx, %eax
	movl	-28(%rbp), %ecx
	addl	%ecx, %eax
	movl	%eax, %edx
	cvtsi2sdq	%rdx, %xmm1
	movsd	LCPI4_0(%rip), %xmm0    ## xmm0 = mem[0],zero
	callq	_pow
	cvttsd2si	%xmm0, %rdx
	movq	%rdx, -80(%rbp)
	movq	$0, -88(%rbp)
	movq	$1, -72(%rbp)
LBB4_6:                                 ##   Parent Loop BB4_4 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	-72(%rbp), %rax
	cmpq	-80(%rbp), %rax
	jg	LBB4_9
## BB#7:                                ##   in Loop: Header=BB4_6 Depth=2
	movl	-24(%rbp), %edi
	callq	_BottomUpTree
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rdi
	callq	_ItemCheck
	addq	-88(%rbp), %rax
	movq	%rax, -88(%rbp)
	movq	-64(%rbp), %rdi
	callq	_DeleteTree
## BB#8:                                ##   in Loop: Header=BB4_6 Depth=2
	movq	-72(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -72(%rbp)
	jmp	LBB4_6
LBB4_9:                                 ##   in Loop: Header=BB4_4 Depth=1
	leaq	L_.str.1(%rip), %rdi
	movq	-80(%rbp), %rsi
	movl	-24(%rbp), %edx
	movq	-88(%rbp), %rcx
	movb	$0, %al
	callq	_printf
	movl	%eax, -100(%rbp)        ## 4-byte Spill
## BB#10:                               ##   in Loop: Header=BB4_4 Depth=1
	movl	-24(%rbp), %eax
	addl	$2, %eax
	movl	%eax, -24(%rbp)
	jmp	LBB4_4
LBB4_11:
	movl	-32(%rbp), %esi
	movq	-56(%rbp), %rdi
	movl	%esi, -104(%rbp)        ## 4-byte Spill
	callq	_ItemCheck
	leaq	L_.str.2(%rip), %rdi
	movl	-104(%rbp), %esi        ## 4-byte Reload
	movq	%rax, %rdx
	movb	$0, %al
	callq	_printf
	xorl	%esi, %esi
	movl	%eax, -108(%rbp)        ## 4-byte Spill
	movl	%esi, %eax
	addq	$112, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"stretch tree of depth %u\t check: %li\n"

L_.str.1:                               ## @.str.1
	.asciz	"%li\t trees of depth %u\t check: %li\n"

L_.str.2:                               ## @.str.2
	.asciz	"long lived tree of depth %u\t check: %li\n"


.subsections_via_symbols
