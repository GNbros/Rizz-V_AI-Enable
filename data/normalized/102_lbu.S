#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_LD_OP( 2, lbu, 0x00000000000000ff, 0,  tdat );
    TEST_LD_OP( 3, lbu, 0x0000000000000000, 1,  tdat );
    TEST_LD_OP( 4, lbu, 0x00000000000000f0, 2,  tdat );
    TEST_LD_OP( 5, lbu, 0x000000000000000f, 3, tdat );

    TEST_LD_OP( 6, lbu, 0x00000000000000ff, -3, tdat4 );
    TEST_LD_OP( 7, lbu, 0x0000000000000000, -2,  tdat4 );
    TEST_LD_OP( 8, lbu, 0x00000000000000f0, -1,  tdat4 );
    TEST_LD_OP( 9, lbu, 0x000000000000000f, 0,   tdat4 );

    TEST_CASE( 10, t0, 0x00000000000000ff, \
    la  ra, tdat; \
    addi ra, ra, -32; \
    lbu t0, 32(ra); \
    )

    TEST_CASE( 11, t0, 0x0000000000000000, \
    la  ra, tdat; \
    addi ra, ra, -6; \
    lbu t0, 7(ra); \
    )

    TEST_LD_DEST_BYPASS( 12, 0, lbu, 0x00000000000000f0, 1, tdat2 );
    TEST_LD_DEST_BYPASS( 13, 1, lbu, 0x000000000000000f, 1, tdat3 );
    TEST_LD_DEST_BYPASS( 14, 2, lbu, 0x0000000000000000, 1, tdat1 );

    TEST_LD_SRC1_BYPASS( 15, 0, lbu, 0x00000000000000f0, 1, tdat2 );
    TEST_LD_SRC1_BYPASS( 16, 1, lbu, 0x000000000000000f, 1, tdat3 );
    TEST_LD_SRC1_BYPASS( 17, 2, lbu, 0x0000000000000000, 1, tdat1 );

    TEST_CASE( 18, sp, 2, \
    la  t0, tdat; \
    lbu  sp, 0(t0); \
    li  sp, 2; \
    )

    TEST_CASE( 19, sp, 2, \
    la  t0, tdat; \
    lbu  sp, 0(t0); \
    nop; \
    li  sp, 2; \
    )

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

tdat:
tdat1:  .byte 0xff
tdat2:  .byte 0x00
tdat3:  .byte 0xf0
tdat4:  .byte 0x0f

RVTEST_DATA_END
