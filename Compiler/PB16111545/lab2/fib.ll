define i32 @fib(i32)  { ;fib function
; <label>:1:
  %2 = alloca i32, align 4;return value

  %3 = icmp eq i32 %0, 0	;n==0?
  br i1 %3, label %4, label %5

; <label>:4:                                      
  store i32 0, i32* %2, align 4;n==0
  br label %14

; <label>:5:                                     
  %6 = icmp eq i32 %0, 1	;n==1?
  br i1 %6, label %7, label %8

; <label>:7:                                     
  store i32 1, i32* %2, align 4 ;n==1
  br label %14

; <label>:8:                                     
  %9 = sub nsw i32 %0, 1	;=n-1
  %10 = call i32 @fib(i32 %9)	;=fib(n-1)

  %11 = sub nsw i32 %0, 2	;=n-2
  %12 = call i32 @fib(i32 %11)	;=fib(n-2)

  %13 = add nsw i32 %10, %12	;=fib(n-1)+fib(n-2)
  store i32 %13, i32* %2, align 4;=result
  br label %14

; <label>:14:                                    
  %15 = load i32, i32* %2, align 4;=return 
  ret i32 %15
}

define i32 @main() {
; <label>:0:
  %1 = alloca i32, align 4 ;return value
  store i32 0, i32* %1, align 4

  %2 = alloca i32, align 4	;%2,x
  store i32 0, i32* %2, align 4

  %3 = alloca i32, align 4		;%3:i
  store i32 0, i32* %3, align 4 ;%3:i=0
  br label %4

; <label>:4:                                      
  %5 = load i32, i32* %3, align 4	;%5=%3:i
  %6 = icmp slt i32 %5, 8	;%6=(%5:i<8)?
  br i1 %6, label %7, label %15	;%6=true-->label 7 else -->label15

; <label>:7:                                      
  %8 = load i32, i32* %3, align 4 ;%8=%3:i
  %9 = call i32 @fib(i32 %8)	;%9=fib(%3:i)
  %10 = load i32, i32* %2, align 4	;%10=%2:x
  %11 = add nsw i32 %10, %9		
  store i32 %11, i32* %2, align 4
  br label %12

; <label>:12:                                    
  %13 = load i32, i32* %3, align 4 ;%13=i
  %14 = add nsw i32 %13, 1 ;%13++
  store i32 %14, i32* %3, align 4 ;%i=%13
  br label %4

; <label>:15:                                     
  %16 = load i32, i32* %2, align 4	;%16=x
  ret i32 %16
}
