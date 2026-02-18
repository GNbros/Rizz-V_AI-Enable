#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_RR_OP( 2,  sh3add, 0x00000000, 0x00000000, 0x00000000 );
    TEST_RR_OP( 3,  sh3add, 0x00000009, 0x00000001, 0x00000001 );
    TEST_RR_OP( 4,  sh3add, 0x0000001f, 0x00000003, 0x00000007 );

    TEST_RR_OP( 5,  sh3add, 0xffffffffffff8000, 0x0000000000000000, 0xffffffffffff8000 );
    TEST_RR_OP( 6,  sh3add, 0xfffffffc00000000, 0xffffffff80000000, 0x00000000 );
    TEST_RR_OP( 7,  sh3add, 0xfffffffbffff8000, 0xffffffff80000000, 0xffffffffffff8000 );

    TEST_RR_OP( 8,  sh3add, 0x0000000000007fff, 0x0000000000000000, 0x0000000000007fff );
    TEST_RR_OP( 9,  sh3add, 0x00000003fffffff8, 0x000000007fffffff, 0x0000000000000000 );
    TEST_RR_OP( 10, sh3add, 0x0000000400007ff7, 0x000000007fffffff, 0x0000000000007fff );

    TEST_RR_OP( 11, sh3add, 0xfffffffc00007fff, 0xffffffff80000000, 0x0000000000007fff );
    TEST_RR_OP( 12, sh3add, 0x00000003ffff7ff8, 0x000000007fffffff, 0xffffffffffff8000 );

    TEST_RR_OP( 13, sh3add, 0xffffffffffffffff, 0x0000000000000000, 0xffffffffffffffff );
    TEST_RR_OP( 14, sh3add, 0xfffffffffffffff9, 0xffffffffffffffff, 0x0000000000000001 );
    TEST_RR_OP( 15, sh3add, 0xfffffffffffffff7, 0xffffffffffffffff, 0xffffffffffffffff );

    TEST_RR_OP( 16, sh3add, 0x0000000080000007, 0x0000000000000001, 0x000000007fffffff );

    TEST_RR_SRC1_EQ_DEST( 17, sh3add, 115, 13, 11 );
    TEST_RR_SRC2_EQ_DEST( 18, sh3add, 123, 14, 11 );
    TEST_RR_SRC12_EQ_DEST( 19, sh3add, 117, 13 );

    TEST_RR_DEST_BYPASS( 20, 0, sh3add, 115, 13, 11 );
    TEST_RR_DEST_BYPASS( 21, 1, sh3add, 123, 14, 11 );
    TEST_RR_DEST_BYPASS( 22, 2, sh3add, 131, 15, 11 );

    TEST_RR_SRC12_BYPASS( 23, 0, 0, sh3add, 115, 13, 11 );
    TEST_RR_SRC12_BYPASS( 24, 0, 1, sh3add, 123, 14, 11 );
    TEST_RR_SRC12_BYPASS( 25, 0, 2, sh3add, 131, 15, 11 );
    TEST_RR_SRC12_BYPASS( 26, 1, 0, sh3add, 115, 13, 11 );
    TEST_RR_SRC12_BYPASS( 27, 1, 1, sh3add, 123, 14, 11 );
    TEST_RR_SRC12_BYPASS( 28, 2, 0, sh3add, 131, 15, 11 );

    TEST_RR_SRC21_BYPASS( 29, 0, 0, sh3add, 115, 13, 11 );
    TEST_RR_SRC21_BYPASS( 30, 0, 1, sh3add, 123, 14, 11 );
    TEST_RR_SRC21_BYPASS( 31, 0, 2, sh3add, 131, 15, 11 );
    TEST_RR_SRC21_BYPASS( 32, 1, 0, sh3add, 115, 13, 11 );
    TEST_RR_SRC21_BYPASS( 33, 1, 1, sh3add, 123, 14, 11 );
    TEST_RR_SRC21_BYPASS( 34, 2, 0, sh3add, 131, 15, 11 );

    TEST_RR_ZEROSRC1( 35, sh3add, 15, 15 );
    TEST_RR_ZEROSRC2( 36, sh3add, 256, 32 );
    TEST_RR_ZEROSRC12( 37, sh3add, 0 );
    TEST_RR_ZERODEST( 38, sh3add, 16, 30 );

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

RVTEST_DATA_END

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_RR_OP( 2,  sh2add.uw, 0x00000000, 0x00000000, 0x00000000 );
    TEST_RR_OP( 3,  sh2add.uw, 0x00000005, 0x00000001, 0x00000001 );
    TEST_RR_OP( 4,  sh2add.uw, 0x00000013, 0x00000003, 0x00000007 );

    TEST_RR_OP( 5,  sh2add.uw, 0xffffffffffff8000, 0x0000000000000000, 0xffffffffffff8000 );
    TEST_RR_OP( 6,  sh2add.uw, 0x0000000200000000, 0xffffffff80000000, 0x00000000 );
    TEST_RR_OP( 7,  sh2add.uw, 0x00000001ffff8000, 0xffffffff80000000, 0xffffffffffff8000 );

    TEST_RR_OP( 8,  sh2add.uw, 0x0000000000007fff, 0x0000000000000000, 0x0000000000007fff );
    TEST_RR_OP( 9,  sh2add.uw, 0x00000001fffffffc, 0x000000007fffffff, 0x0000000000000000 );
    TEST_RR_OP( 10, sh2add.uw, 0x0000000200007ffb, 0x000000007fffffff, 0x0000000000007fff );

    TEST_RR_OP( 11, sh2add.uw, 0x0000000200007fff, 0xffffffff80000000, 0x0000000000007fff );
    TEST_RR_OP( 12, sh2add.uw, 0x00000001ffff7ffc, 0x000000007fffffff, 0xffffffffffff8000 );

    TEST_RR_OP( 13, sh2add.uw, 0xffffffffffffffff, 0x0000000000000000, 0xffffffffffffffff );
    TEST_RR_OP( 14, sh2add.uw, 0x00000003fffffffd, 0xffffffffffffffff, 0x0000000000000001 );
    TEST_RR_OP( 15, sh2add.uw, 0x00000003fffffffb, 0xffffffffffffffff, 0xffffffffffffffff );

    TEST_RR_OP( 16, sh2add.uw, 0x0000000080000003, 0x0000000000000001, 0x000000007fffffff );

    TEST_RR_SRC1_EQ_DEST( 17, sh2add.uw, 63, 13, 11 );
    TEST_RR_SRC2_EQ_DEST( 18, sh2add.uw, 67, 14, 11 );
    TEST_RR_SRC12_EQ_DEST( 19, sh2add.uw, 65, 13 );

    TEST_RR_DEST_BYPASS( 20, 0, sh2add.uw, 63, 13, 11 );
    TEST_RR_DEST_BYPASS( 21, 1, sh2add.uw, 67, 14, 11 );
    TEST_RR_DEST_BYPASS( 22, 2, sh2add.uw, 71, 15, 11 );

    TEST_RR_SRC12_BYPASS( 23, 0, 0, sh2add.uw, 63, 13, 11 );
    TEST_RR_SRC12_BYPASS( 24, 0, 1, sh2add.uw, 67, 14, 11 );
    TEST_RR_SRC12_BYPASS( 25, 0, 2, sh2add.uw, 71, 15, 11 );
    TEST_RR_SRC12_BYPASS( 26, 1, 0, sh2add.uw, 63, 13, 11 );
    TEST_RR_SRC12_BYPASS( 27, 1, 1, sh2add.uw, 67, 14, 11 );
    TEST_RR_SRC12_BYPASS( 28, 2, 0, sh2add.uw, 71, 15, 11 );

    TEST_RR_SRC21_BYPASS( 29, 0, 0, sh2add.uw, 63, 13, 11 );
    TEST_RR_SRC21_BYPASS( 30, 0, 1, sh2add.uw, 67, 14, 11 );
    TEST_RR_SRC21_BYPASS( 31, 0, 2, sh2add.uw, 71, 15, 11 );
    TEST_RR_SRC21_BYPASS( 32, 1, 0, sh2add.uw, 63, 13, 11 );
    TEST_RR_SRC21_BYPASS( 33, 1, 1, sh2add.uw, 67, 14, 11 );
    TEST_RR_SRC21_BYPASS( 34, 2, 0, sh2add.uw, 71, 15, 11 );

    TEST_RR_ZEROSRC1( 35, sh2add.uw, 15, 15 );
    TEST_RR_ZEROSRC2( 36, sh2add.uw, 128, 32 );
    TEST_RR_ZEROSRC12( 37, sh2add.uw, 0 );
    TEST_RR_ZERODEST( 38, sh2add.uw, 16, 30 );

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

RVTEST_DATA_END
