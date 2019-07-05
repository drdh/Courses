 .data
data_in: .word 1 2 3
data_out: .word 0 0 0
      .text
      la   $t0, data_in
      lw   $t1, 0($t0)		#$t1=1
      lw   $t2, 4($t0)		#$t2=2
      lw   $t3, 8($t0)		#$t3=3

      add  $t1, $t1, $t2	#$t1=3
      add  $t1, $t2, $t3	#$t1=5
      add  $t2, $t1, $t3	#$t2=8
      add  $t3, $t1, $t2	#$t3=13

      la   $t0, data_out
      sw   $t1, 0($t0)
      sw   $t2, 4($t0)
      sw   $t3, 8($t0)
out:  
	j out
