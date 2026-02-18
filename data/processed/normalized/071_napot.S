#include "riscv_test.h"
#include "test_macros.h"

#if (DRAM_BASE >> 30 << 30) != DRAM_BASE
# error This test requires DRAM_BASE be SV39 superpage-aligned
#endif

#if __riscv_xlen != 64
# error This test requires RV64
#endif

RVTEST_RV64M
RVTEST_CODE_BEGIN

#define MY_VA 0x40201010

  la a0, my_data
  srl a0, a0, 12

  li a1, ~0xF
  and a0, a0, a1
  ori a0, a0, 0x8

  sll a0, a0, PTE_PPN_SHIFT
  li a1, PTE_V | PTE_U | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D | PTE_N
  or a0, a0, a1

  la a1, page_table
  addi a1, a1, ((MY_VA >> 12) & 0x1FF) * 8

  sd a0, (a1)

  la a0, page_table
  srl a0, a0, 12
  sll a0, a0, PTE_PPN_SHIFT
  li a1, PTE_V
  or a0, a0, a1

  la a1, page_table
  addi a1, a1, ((MY_VA >> 21) & 0x1FF) * 8
  li a2, 1 << 12
  add a1, a1, a2

  sd a0, (a1)

  la a0, page_table
  li a1, 1 << 12
  add a0, a0, a1
  srl a0, a0, 12
  sll a0, a0, PTE_PPN_SHIFT
  li a1, PTE_V
  or a0, a0, a1

  la a1, page_table
  addi a1, a1, ((MY_VA >> 30) & 0x1FF) * 8
  li a2, 2 << 12
  add a1, a1, a2

  sd a0, (a1)

  la a0, my_data
  li a1, ~0xFFFF
  and a0, a0, a1
  li a1, 0x8000 | (MY_VA & 0xFFF)
  or a3, a0, a1
  li a1, 0
  sw a1, (a3)

  li TESTNUM, 1

  la a1, page_table
  li a2, 2 << 12
  add a1, a1, a2
  srl a1, a1, 12
  li a0, (SATP_MODE & ~(SATP_MODE<<1)) * SATP_MODE_SV39
  or a0, a0, a1
  csrw satp, a0
  sfence.vma

  li a1, ((MSTATUS_MPP & ~(MSTATUS_MPP<<1)) * PRV_S) | MSTATUS_MPRV | MSTATUS_SUM
  csrs mstatus, a1

  li a0, MY_VA
  li a1, 42
napot_store:
  sw a1, (a0)

  li a1, MSTATUS_MPRV
  csrc mstatus, a1

  lw a1, (a3)

  li a0, 42
  beq a1, a0, die

  la a0, my_data
  li a1, MY_VA & 0xFFFF
  add a0, a0, a1
  lw a1, (a0)
  li a2, 42

  bne a1, a2, die

  RVTEST_PASS

  TEST_PASSFAIL

  .align 2
  .global mtvec_handler
mtvec_handler:
  csrr t5, mcause
  li t6, CAUSE_STORE_PAGE_FAULT
  bne t5, t6, die
  csrr t5, mepc
  la t6, napot_store
  bne t5, t6, die
  csrr t5, mtval
  li t6, MY_VA
  beq t5, t6, pass
die:
  RVTEST_FAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

.align 20
page_table: .dword 0

.align 20
my_data: .dword 0

RVTEST_DATA_END

#include "riscv_test.h"
#undef RVTEST_RV64UF
#define RVTEST_RV64UF RVTEST_RV32UF

#include "../rv64uzfh/fcvt.S"

#include "riscv_test.h"
#undef RVTEST_RV64UH
#define RVTEST_RV64UH RVTEST_RV32UH

#include "../rv64uzfh/fdiv.S"
