#*****************************************************************************
# orn.S
#-----------------------------------------------------------------------------
#
# Test orn instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Logical tests
  #-------------------------------------------------------------

  TEST_RR_OP( 2, orn, 0xfffffffffff0fff0, 0xffffffffff00ff00, 0x000000000f0f0f0f );
  TEST_RR_OP( 3, orn, 0x000000000fff0fff, 0x000000000ff00ff0, 0xfffffffff0f0f0f0 );
  TEST_RR_OP( 4, orn, 0xfffffffff0fff0ff, 0x0000000000ff00ff, 0x000000000f0f0f0f );
  TEST_RR_OP( 5, orn, 0xffffffffff0fff0f, 0xfffffffff00ff00f, 0xfffffffff0f0f0f0 );

#if __riscv_xlen == 64
  TEST_RR_OP( 50, orn, 0x0fff0fff0fff0fff, 0x0ff00ff00ff00ff0, 0xf0f0f0f0f0f0f0f0 );
  TEST_RR_OP( 51, orn, 0xf0fff0fff0fff0ff, 0x00ff00ff00ff00ff, 0x0f0f0f0f0f0f0f0f );
  TEST_RR_OP( 52, orn, 0xff0fff0fff0fff0f, 0xf00ff00ff00ff00f, 0xf0f0f0f0f0f0f0f0 );
#endif

  #-------------------------------------------------------------
  # Source/Destination tests
  #-------------------------------------------------------------

  TEST_RR_SRC1_EQ_DEST( 6, orn, 0xfffffffffff0fff0, 0xffffffffff00ff00, 0x000000000f0f0f0f );
  TEST_RR_SRC2_EQ_DEST( 7, orn, 0xfffffffffff0fff0, 0xffffffffff00ff00, 0x000000000f0f0f0f );
  TEST_RR_SRC12_EQ_DEST( 8, orn, 0xffffffffffffffff, 0xffffffffff00ff00 );

  #-------------------------------------------------------------
  # Bypassing tests
  #-------------------------------------------------------------

  TEST_RR_DEST_BYPASS( 9,  0, orn, 0xfffffffffff0fff0, 0xffffffffff00ff00, 0x000000000f0f0f0f );
  TEST_RR_DEST_BYPASS( 10, 1, orn, 0x000000000fff0fff, 0x000000000ff00ff0, 0xfffffffff0f0f0f0 );
  TEST_RR_DEST_BYPASS( 11, 2, orn, 0xfffffffff0fff0ff, 0x0000000000ff00ff, 0x000000000f0f0f0f );

  TEST_RR_SRC12_BYPASS( 12, 0, 0, orn, 0xfffffffffff0fff0, 0xffffffffff00ff00, 0x000000000f0f0f0f );
  TEST_RR_SRC12_BYPASS( 13, 0, 1, orn, 0x000000000fff0fff, 0x000000000ff00ff0, 0xfffffffff0f0f0f0 );
  TEST_RR_SRC12_BYPASS( 14, 0, 2, orn, 0xfffffffff0fff0ff, 0x0000000000ff00ff, 0x000000000f0f0f0f );
  TEST_RR_SRC12_BYPASS( 15, 1, 0, orn, 0xfffffffffff0fff0, 0xffffffffff00ff00, 0x000000000f0f0f0f );
  TEST_RR_SRC12_BYPASS( 16, 1, 1, orn, 0x000000000fff0fff, 0x000000000ff00ff0, 0xfffffffff0f0f0f0 );
  TEST_RR_SRC12_BYPASS( 17, 2, 0, orn, 0xfffffffff0fff0ff, 0x0000000000ff00ff, 0x000000000f0f0f0f );

  TEST_RR_SRC21_BYPASS( 18, 0, 0, orn, 0xfffffffffff0fff0, 0xffffffffff00ff00, 0x000000000f0f0f0f );
  TEST_RR_SRC21_BYPASS( 19, 0, 1, orn, 0x000000000fff0fff, 0x000000000ff00ff0, 0xfffffffff0f0f0f0 );
  TEST_RR_SRC21_BYPASS( 20, 0, 2, orn, 0xfffffffff0fff0ff, 0x0000000000ff00ff, 0x000000000f0f0f0f );
  TEST_RR_SRC21_BYPASS( 21, 1, 0, orn, 0xfffffffffff0fff0, 0xffffffffff00ff00, 0x000000000f0f0f0f );
  TEST_RR_SRC21_BYPASS( 22, 1, 1, orn, 0x000000000fff0fff, 0x000000000ff00ff0, 0xfffffffff0f0f0f0 );
  TEST_RR_SRC21_BYPASS( 23, 2, 0, orn, 0xfffffffff0fff0ff, 0x0000000000ff00ff, 0x000000000f0f0f0f );

  TEST_RR_ZEROSRC1( 24, orn, 0x0000000000ff00ff, 0xffffffffff00ff00 );
  TEST_RR_ZEROSRC2( 25, orn, -1, 0x0000000000ff00ff );
  TEST_RR_ZEROSRC12( 26, orn, -1 );
  TEST_RR_ZERODEST( 27, orn, 0x0000000011111111, 0x0000000022222222 );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
# See LICENSE for license details.

#*****************************************************************************
# orc.b.S
#-----------------------------------------------------------------------------
#
# Test orc.b instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Arithmetic tests
  #-------------------------------------------------------------

  TEST_R_OP( 2,  orc.b, 0x0000000000000000, 0x0000000000000000);
  TEST_R_OP( 3,  orc.b, 0x00000000000000ff, 0x0000000000000001);
  TEST_R_OP( 4,  orc.b, 0x00000000000000ff, 0x0000000000000003);

  TEST_R_OP( 5,  orc.b, 0xffffffffffffff00, 0xffffffffffff8000 );
  TEST_R_OP( 6,  orc.b, 0x0000000000ff0000, 0x0000000000800000 );
  TEST_R_OP( 7,  orc.b, 0x00ffffffffffff00, 0x0004ffffffff8000 );

  TEST_R_OP( 8,  orc.b, 0x000000000000ffff, 0x0000000000007fff);
  TEST_R_OP( 9,  orc.b, 0x00000000ffffffff, 0x000000007fffffff);
  TEST_R_OP( 10, orc.b, 0x0000000000ffffff, 0x000000000007ffff );

  TEST_R_OP( 11, orc.b, 0xffffffffff000000, 0xffffffff80000000);
  TEST_R_OP( 12, orc.b, 0x00ffffffffffff00, 0x00ff578f121f5000);

  TEST_R_OP( 13, orc.b, 0xff00000000000000, 0x8000000000000000);
  TEST_R_OP( 14, orc.b, 0x00000000000000ff, 0x000000000000000e);
  TEST_R_OP( 15, orc.b, 0xff0000ffffffffff, 0xa000000320401341);

  #-------------------------------------------------------------
  # Source/Destination tests
  #-------------------------------------------------------------

  TEST_R_SRC1_EQ_DEST( 16, orc.b, 0xff, 13);
  TEST_R_SRC1_EQ_DEST( 17, orc.b, 0xff, 11);

  #-------------------------------------------------------------
  # Bypassing tests
  #-------------------------------------------------------------

  TEST_R_DEST_BYPASS( 18, 0, orc.b, 0xff, 13);
  TEST_R_DEST_BYPASS( 29, 1, orc.b, 0xff, 19);
  TEST_R_DEST_BYPASS( 20, 2, orc.b, 0xff, 34);

  #-------------------------------------------------------------
  # Other tests
  #-------------------------------------------------------------

  TEST_R_OP( 21,  orc.b, 0x0000000000ffff00, 0x00000000007f8000 );
  TEST_R_OP( 22,  orc.b, 0x0000000000ffff00, 0x0000000000808000 );
  TEST_R_OP( 23,  orc.b, 0x00000000ffffff00, 0x0000000001808000 );

  TEST_R_OP( 24,  orc.b, 0x000000ff0000ffff, 0x0000000300007fff);
  TEST_R_OP( 25,  orc.b, 0x000000ffffffffff, 0x000000077fffffff);
  TEST_R_OP( 26,  orc.b, 0x000000ff00ffffff, 0x0000000f0007ffff);

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
# See LICENSE for license details.

