# RISC-V Assembly Smoke Test Sample B
.text
.globl fib
.type fib, @function
fib:
    li a0, 0
    li a1, 1
    mv a2, a0
    ret
