#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

  TEST_LD_OP( 2, lh, 0x00000000000000ff, 0,  tdat );
  TEST_LD_OP( 3, lh, 0xffffffffffffff00, 2,  tdat );
  TEST_LD_OP( 4, lh, 0x0000000000000ff0, 4,  tdat );
  TEST_LD_OP( 5, lh, 0xfffffffffffff00f, 6, tdat );

  TEST_LD_OP( 6, lh, 0x00000000000000ff, -6,  tdat4 );
  TEST_LD_OP( 7, lh, 0xffffffffffffff00, -4,  tdat4 );
  TEST_LD_OP( 8, lh, 0x0000000000000ff0, -2,  tdat4 );
  TEST_LD_OP( 9, lh, 0xfffffffffffff00f,  0, tdat4 );

  TEST_CASE( 10, t0, 0x00000000000000ff, \
  la  ra, tdat; \
  addi ra, ra, -32; \
  lh t0, 32(ra); \
  )

  TEST_CASE( 11, t0, 0xffffffffffffff00, \
  la  ra, tdat; \
  addi ra, ra, -5; \
  lh t0, 7(ra); \
  )

  TEST_LD_DEST_BYPASS( 12, 0, lh, 0x0000000000000ff0, 2, tdat2 );
  TEST_LD_DEST_BYPASS( 13, 1, lh, 0xfffffffffffff00f, 2, tdat3 );
  TEST_LD_DEST_BYPASS( 14, 2, lh, 0xffffffffffffff00, 2, tdat1 );

  TEST_LD_SRC1_BYPASS( 15, 0, lh, 0x0000000000000ff0, 2, tdat2 );
  TEST_LD_SRC1_BYPASS( 16, 1, lh, 0xfffffffffffff00f, 2, tdat3 );
  TEST_LD_SRC1_BYPASS( 17, 2, lh, 0xffffffffffffff00, 2, tdat1 );

  TEST_CASE( 18, sp, 2, \
  la  t0, tdat; \
  lh  sp, 0(t0); \
  li  sp, 2; \
  )

  TEST_CASE( 19, sp, 2, \
  la  t0, tdat; \
  lh  sp, 0(t0); \
  nop; \
  li  sp, 2; \
  )

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
tdat1:  .half 0x00ff
tdat2:  .half 0xff00
tdat3:  .half 0x0ff0
tdat4:  .half 0xf00f

RVTEST_DATA_END
