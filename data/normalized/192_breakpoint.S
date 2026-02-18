#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

    li TESTNUM, 2

    la a0, 1f
    csrrw a0, mtvec, a0
    li a1, 0x8
    csrs tcontrol, a1
.p2align 2
1:
    csrw mtvec, a0

    csrsi mstatus, MSTATUS_MIE

    csrw tselect, zero
    csrr a1, tselect
    bne zero, a1, pass

    la a2, 1f
    csrw tdata2, a2
    li a0, (2 << (__riscv_xlen - 4)) | MCONTROL_M | MCONTROL_EXECUTE
    csrw tdata1, a0
    csrr a1, tdata1
    bne a0, a1, 2f
    .align 2
1:
    beqz zero, fail

    li TESTNUM, 3
    lw a0, (a2)

2:
    li TESTNUM, 4
    li a0, (2 << (__riscv_xlen - 4)) | MCONTROL_M | MCONTROL_LOAD
    csrw tdata1, a0
    csrr a1, tdata1
    bne a0, a1, 2f
    la a2, data1
    csrw tdata2, a2

    lw a2, (a2)
    beqz a2, fail

    li TESTNUM, 5
    sw zero, (a2)

2:
    li TESTNUM, 6
    li a0, (2 << (__riscv_xlen - 4)) |  MCONTROL_M | MCONTROL_STORE
    csrw tdata1, a0
    csrr a1, tdata1
    bne a0, a1, 2f

    sw a2, (a2)

    li TESTNUM, 7
    lw a2, (a2)
    bnez a2, fail

    li a0, 1
    csrw tselect, a0
    csrr a1, tselect
    bne a0, a1, pass

    li a0, (2 << (__riscv_xlen - 4)) | MCONTROL_M | MCONTROL_LOAD
    csrw tdata1, a0
    la a3, data2
    csrw tdata2, a3

    li TESTNUM, 8
    lw a3, (a3)
    beqz a3, fail

    li TESTNUM, 10
    la a2, data1
    sw a2, (a2)
    li TESTNUM, 11
    lw a2, (a2)
    bnez a2, fail

2:
    TEST_PASSFAIL

    .align 2
    .global mtvec_handler
mtvec_handler:
    andi t0, TESTNUM, 1
    bnez t0, fail

    li t0, CAUSE_BREAKPOINT
    csrr t1, mcause
    bne t0, t1, fail

    csrr t0, mepc
    addi t0, t0, 4
    csrw mepc, t0
    mret

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

data1: .word 0
data2: .word 0

RVTEST_DATA_END
