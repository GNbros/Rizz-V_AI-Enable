#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  TEST_RR_OP( 2,  rol, 0x00000001, 0x00000001, 0  );
  TEST_RR_OP( 3,  rol, 0x00000002, 0x00000001, 1  );
  TEST_RR_OP( 4,  rol, 0x00000080, 0x00000001, 7  );
  TEST_RR_OP( 5,  rol, 0x00004000, 0x00000001, 14 );
  TEST_RR_OP( 6,  rol, 0x80000000, 0x00000001, 31 );

  TEST_RR_OP( 7,  rol, 0xffffffff, 0xffffffff, 0  );
  TEST_RR_OP( 8,  rol, 0xffffffff, 0xffffffff, 1  );
  TEST_RR_OP( 9,  rol, 0xffffffff, 0xffffffff, 7  );
  TEST_RR_OP( 10, rol, 0xffffffff, 0xffffffff, 14 );
  TEST_RR_OP( 11, rol, 0xffffffff, 0xffffffff, 31 );

  TEST_RR_OP( 12, rol, 0x21212121, 0x21212121, 0  );
  TEST_RR_OP( 13, rol, 0x42424242, 0x21212121, 1  );
  TEST_RR_OP( 14, rol, 0x90909090, 0x21212121, 7  );
  TEST_RR_OP( 15, rol, 0x48484848, 0x21212121, 14 );
  TEST_RR_OP( 16, rol, 0x90909090, 0x21212121, 31 );

  TEST_RR_OP( 17, rol, 0x21212121, 0x21212121, 0xffffffe0 );
  TEST_RR_OP( 18, rol, 0x42424242, 0x21212121, 0xffffffe1 );
  TEST_RR_OP( 19, rol, 0x90909090, 0x21212121, 0xffffffe7 );
  TEST_RR_OP( 20, rol, 0x48484848, 0x21212121, 0xffffffee );
  TEST_RR_OP( 21, rol, 0x90909090, 0x21212121, 0xffffffff );

  TEST_RR_OP( 44, rol, 0x12345678, 0x12345678, 0 );
  TEST_RR_OP( 45, rol, 0x23456781, 0x12345678, 4 );
  TEST_RR_OP( 46, rol, 0x92345678, 0x92345678, 0 );
  TEST_RR_OP( 47, rol, 0x93456789, 0x99345678, 4 );

  TEST_RR_SRC1_EQ_DEST( 22, rol, 0x00000080, 0x00000001, 7  );
  TEST_RR_SRC2_EQ_DEST( 23, rol, 0x00004000, 0x00000001, 14 );
  TEST_RR_SRC12_EQ_DEST( 24, rol, 24, 3 );

  TEST_RR_DEST_BYPASS( 25, 0, rol, 0x00000080, 0x00000001, 7  );
  TEST_RR_DEST_BYPASS( 26, 1, rol, 0x00004000, 0x00000001, 14 );
  TEST_RR_DEST_BYPASS( 27, 2, rol, 0x80000000, 0x00000001, 31 );

  TEST_RR_SRC12_BYPASS( 28, 0, 0, rol, 0x00000080, 0x00000001, 7  );
  TEST_RR_SRC12_BYPASS( 29, 0, 1, rol, 0x00004000, 0x00000001, 14 );
  TEST_RR_SRC12_BYPASS( 30, 0, 2, rol, 0x80000000, 0x00000001, 31 );
  TEST_RR_SRC12_BYPASS( 31, 1, 0, rol, 0x00000080, 0x00000001, 7  );
  TEST_RR_SRC12_BYPASS( 32, 1, 1, rol, 0x00004000, 0x00000001, 14 );
  TEST_RR_SRC12_BYPASS( 33, 2, 0, rol, 0x80000000, 0x00000001, 31 );

  TEST_RR_SRC21_BYPASS( 34, 0, 0, rol, 0x00000080, 0x00000001, 7  );
  TEST_RR_SRC21_BYPASS( 35, 0, 1, rol, 0x00004000, 0x00000001, 14 );
  TEST_RR_SRC21_BYPASS( 36, 0, 2, rol, 0x80000000, 0x00000001, 31 );
  TEST_RR_SRC21_BYPASS( 37, 1, 0, rol, 0x00000080, 0x00000001, 7  );
  TEST_RR_SRC21_BYPASS( 38, 1, 1, rol, 0x00004000, 0x00000001, 14 );
  TEST_RR_SRC21_BYPASS( 39, 2, 0, rol, 0x80000000, 0x00000001, 31 );

  TEST_RR_ZEROSRC1( 40, rol, 0, 15 );
  TEST_RR_ZEROSRC2( 41, rol, 32, 32 );
  TEST_RR_ZEROSRC12( 42, rol, 0 );
  TEST_RR_ZERODEST( 43, rol, 1024, 2048 );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64uzbb/max.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64uzbb/maxu.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64uzbb/orn.S"

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  TEST_R_OP( 2,  orc.b, 0x00000000, 0x00000000);
  TEST_R_OP( 3,  orc.b, 0x000000ff, 0x00000001);
  TEST_R_OP( 4,  orc.b, 0x000000ff, 0x00000003);

  TEST_R_OP( 5,  orc.b, 0xffffff00, 0xffff8000 );
  TEST_R_OP( 6,  orc.b, 0x00ff0000, 0x00800000 );
  TEST_R_OP( 7,  orc.b, 0xffffff00, 0xffff8000 );

  TEST_R_OP( 8,  orc.b, 0x0000ffff, 0x00007fff);
  TEST_R_OP( 9,  orc.b, 0xffffffff, 0x7fffffff);
  TEST_R_OP( 10, orc.b, 0x00ffffff, 0x0007ffff );

  TEST_R_OP( 11, orc.b, 0xff000000, 0x80000000);
  TEST_R_OP( 12, orc.b, 0xffffff00, 0x121f5000);

  TEST_R_OP( 13, orc.b, 0x00000000, 0x00000000);
  TEST_R_OP( 14, orc.b, 0x000000ff, 0x0000000e);
  TEST_R_OP( 15, orc.b, 0xffffffff, 0x20401341);

  TEST_R_SRC1_EQ_DEST( 16, orc.b, 0xff, 13);
  TEST_R_SRC1_EQ_DEST( 17, orc.b, 0xff, 11);

  TEST_R_DEST_BYPASS( 18, 0, orc.b, 0xff, 13);
  TEST_R_DEST_BYPASS( 29, 1, orc.b, 0xff, 19);
  TEST_R_DEST_BYPASS( 20, 2, orc.b, 0xff, 34);

  TEST_R_OP( 21,  orc.b, 0x00ffff00, 0x007f8000 );
  TEST_R_OP( 22,  orc.b, 0x00ffff00, 0x00808000 );
  TEST_R_OP( 23,  orc.b, 0xffffff00, 0x01808000 );

  TEST_R_OP( 24,  orc.b, 0x0000ffff, 0x00007fff);
  TEST_R_OP( 25,  orc.b, 0xffffffff, 0x7fffffff);
  TEST_R_OP( 26,  orc.b, 0x00ffffff, 0x0007ffff);

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64uzbb/sext_h.S"
