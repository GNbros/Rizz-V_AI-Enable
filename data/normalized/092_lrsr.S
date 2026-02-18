#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

la a0, coreid
li a1, 1
amoadd.w a2, a1, (a0)

1:li a3, 1
bgeu a2, a3, 1b

1: lw a1, (a0)
bltu a1, a3, 1b

TEST_CASE( 2, a4, 1, \
    la a0, foo; \
    li a5, 0xdeadbeef; \
    sc.w a4, a5, (a0); \
)

TEST_CASE( 3, a4, 0, \
    lw a4, foo; \
)

#define LOG_ITERATIONS 10

la a0, foo
li a1, 1<<LOG_ITERATIONS
addi a2, a2, 1
1: lr.w a4, (a0)
add a4, a4, a2
sc.w a4, a4, (a0)
bnez a4, 1b
addi a1, a1, -1
bnez a1, 1b

la a0, barrier
li a1, 1
amoadd.w zero, a1, (a0)
1: lw a1, (a0)
blt a1, a3, 1b
fence

TEST_CASE( 5, a0, 0, \
    lw a0, foo; \
    slli a1, a3, LOG_ITERATIONS-1; \
1:sub a0, a0, a1; \
    addi a3, a3, -1; \
    bgez a3, 1b
)

TEST_CASE( 6, a1, 2, \
    la a0, foo; \
1:lr.w a1, (a0); \
    sc.w a1, zero, (a0); \
    bnez a1, 1b; \
    sc.w a1, zero, (a0); \
    \
    sc.w a2, zero, (a0); \
    add a1, a1, a2
)

TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

coreid: .word 0
barrier: .word 0
foo: .word 0
.skip 1024
fooTest3: .word 0
RVTEST_DATA_END
