#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_CASE( 2, ra, 0x0000000000000000, lui ra, 0x00000 );
    TEST_CASE( 3, ra, 0xfffffffffffff800, lui ra, 0xfffff;sra ra,ra,1);
    TEST_CASE( 4, ra, 0x00000000000007ff, lui ra, 0x7ffff;sra ra,ra,20);
    TEST_CASE( 5, ra, 0xfffffffffffff800, lui ra, 0x80000;sra ra,ra,20);

    TEST_CASE( 6, zero, 0, lui zero, 0x80000 );

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

RVTEST_DATA_END
