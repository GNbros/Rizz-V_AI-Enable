#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

  li TESTNUM, 2

  la t0, page_table_3
  srl t0, t0, RISCV_PGSHIFT - PTE_PPN_SHIFT
  ori t0, t0, PTE_V
  sd t0, page_table_2, t1

  la t0, page_table_2
  srl t0, t0, RISCV_PGSHIFT - PTE_PPN_SHIFT
  ori t0, t0, PTE_V
  sd t0, page_table_1, t1

  la t0, code_page_1
  srl t0, t0, RISCV_PGSHIFT - PTE_PPN_SHIFT
  ori t0, t0, PTE_V | PTE_X | PTE_A
  sd t0, page_table_3 + 8, t1

  la t0, code_page_2
  srl t0, t0, RISCV_PGSHIFT - PTE_PPN_SHIFT
  ori t0, t0, PTE_V | PTE_X | PTE_A
  sd t0, page_table_3 + 0, t1

  li a0, (SATP_MODE & ~(SATP_MODE<<1)) * SATP_MODE_SV39
  la a1, page_table_1
  srl a1, a1, RISCV_PGSHIFT
  or a1, a1, a0
  csrw satp, a1
  sfence.vma

  la a2, 1f
  csrwi mepc, 0
  li a1, ((MSTATUS_MPP & ~(MSTATUS_MPP<<1)) * PRV_S)
  csrs mstatus, a1
  mret

1:
  li TESTNUM, 2
  addi a0, a0, -321
  bnez a0, fail

  li TESTNUM, 3
  la a2, 1f
  li t0, RISCV_PGSIZE
  csrw mepc, t0
  mret

1:
  addi a0, a0, -123
  bnez a0, fail

  li TESTNUM, 4
  la a2, 1f
  csrwi mepc, 0
  mret

  .align 2
1:
  addi a0, a0, -321
  bnez a0, fail

  li TESTNUM, 5

  la t0, code_page_1
  srl t0, t0, RISCV_PGSHIFT - PTE_PPN_SHIFT
  ori t0, t0, PTE_V | PTE_X | PTE_A
  sd t0, page_table_3 + 0, t1
  sfence.vma

  la a2, 1f
  csrwi mepc, 0
  mret

  .align 2
1:
  addi a0, a0, -123
  bnez a0, fail

  RVTEST_PASS

  TEST_PASSFAIL

  .align 2
  .global mtvec_handler
mtvec_handler:
  csrr t0, mcause
  add t0, t0, -CAUSE_STORE_PAGE_FAULT
  bnez t0, fail

  jr a2

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

.align 12
page_table_1: .dword 0
.align 12
page_table_2: .dword 0
.align 12
page_table_3: .dword 0
.align 13
code_page_1:
  li a0, 123
  sw zero, (zero)
.align 12
code_page_2:
  li a0, 321
  sw zero, (zero)

RVTEST_DATA_END
