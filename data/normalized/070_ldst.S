#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32UF
RVTEST_CODE_BEGIN

    TEST_CASE(2, a0, 0x40000000, la a1, tdat; flw f1, 4(a1); fsw f1, 20(a1); lw a0, 20(a1))
    TEST_CASE(3, a0, 0xbf800000, la a1, tdat; flw f1, 0(a1); fsw f1, 24(a1); lw a0, 24(a1))

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

tdat:
.word 0xbf800000
.word 0x40000000
.word 0x40400000
.word 0xc0800000
.word 0xdeadbeef
.word 0xcafebabe
.word 0xabad1dea
.word 0x1337d00d

RVTEST_DATA_END

#include "riscv_test.h"
#undef RVTEST_RV64UF
#define RVTEST_RV64UF RVTEST_RV32UF

#include "../rv64uf/fmin.S"

#include "riscv_test.h"
#undef RVTEST_RV64UF
#define RVTEST_RV64UF RVTEST_RV32UF

#include "../rv64uf/recoding.S"

#include "riscv_test.h"
#undef RVTEST_RV64UF
#define RVTEST_RV64UF RVTEST_RV32UF

#include "../rv64uf/fclass.S"

#include "riscv_test.h"
#undef RVTEST_RV64UF
#define RVTEST_RV64UF RVTEST_RV32UF

#include "../rv64uf/fcmp.S"

#include "riscv_test.h"
#undef RVTEST_RV64UF
#define RVTEST_RV64UF RVTEST_RV32UF

#include "../rv64uf/fadd.S"

#include "riscv_test.h"
#undef RVTEST_RV64UF
#define RVTEST_RV64UF RVTEST_RV32UF

#include "../rv64uf/move.S"

#include "riscv_test.h"
#undef RVTEST_RV64UF
#define RVTEST_RV64UF RVTEST_RV32UF

#include "../rv64uf/fcvt_w.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/amoor_w.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/amomax_w.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/amoswap_w.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/lrsc.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/amomaxu_w.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/amoand_w.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/amoxor_w.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/amominu_w.S"

#include "riscv_test.h"
#undef RVTEST_RV64U
#define RVTEST_RV64U RVTEST_RV32U

#include "../rv64ua/amoadd_w.S"
