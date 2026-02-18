#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

    TEST_CASE( 2, zero, 0, csrrc  zero, cycle,    zero);
    TEST_CASE( 3, zero, 0, csrrs  zero, cycle,    zero);
    TEST_CASE( 4, zero, 0, csrrci zero, cycle,    0);
    TEST_CASE( 5, zero, 0, csrrsi zero, cycle,    0);
    TEST_CASE( 6, zero, 0, csrrc  zero, instret,  zero);
    TEST_CASE( 7, zero, 0, csrrs  zero, instret,  zero);
    TEST_CASE( 8, zero, 0, csrrci zero, instret,  0);
    TEST_CASE( 9, zero, 0, csrrsi zero, instret,  0);
#if __riscv_xlen == 32
    TEST_CASE(12, zero, 0, csrrc  zero, cycleh,   zero);
    TEST_CASE(13, zero, 0, csrrs  zero, cycleh,   zero);
    TEST_CASE(14, zero, 0, csrrci zero, cycleh,   0);
    TEST_CASE(15, zero, 0, csrrsi zero, cycleh,   0);
    TEST_CASE(16, zero, 0, csrrc  zero, instreth, zero);
    TEST_CASE(17, zero, 0, csrrs  zero, instreth, zero);
    TEST_CASE(18, zero, 0, csrrci zero, instreth, 0);
    TEST_CASE(19, zero, 0, csrrsi zero, instreth, 0);
#endif

2:
    TEST_PASSFAIL

    .align 2
    .global mtvec_handler
mtvec_handler:
    j fail

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

RVTEST_DATA_END
