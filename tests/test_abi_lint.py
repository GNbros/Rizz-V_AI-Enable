"""
Unit tests for scripts/abi_lint.py
===================================
Each test uses an inline .S fixture string – no external files needed.

Run:
    pytest tests/test_abi_lint.py -v
"""

import pytest
from scripts.abi_lint import lint_asm, LintResult, Violation

# ── Fixtures (inline assembly snippets) ─────────────────────────────────────

GOOD_FUNC = """\
.globl my_func
my_func:
  addi sp, sp, -16
  sw ra, 12(sp)
  sw s0, 8(sp)
  addi s0, sp, 16
  li a0, 42
  lw s0, 8(sp)
  lw ra, 12(sp)
  addi sp, sp, 16
  ret
"""

MISSING_RET = """\
.globl bad_func
bad_func:
  addi sp, sp, -16
  sw ra, 12(sp)
  li a0, 1
  lw ra, 12(sp)
  addi sp, sp, 16
  # forgot ret!
"""

UNBALANCED_STACK = """\
.globl stack_bad
stack_bad:
  addi sp, sp, -32
  sw ra, 28(sp)
  li a0, 0
  lw ra, 28(sp)
  addi sp, sp, 16
  ret
"""

MISSING_RA_RESTORE = """\
.globl ra_bad
ra_bad:
  addi sp, sp, -16
  sw ra, 12(sp)
  li a0, 0
  addi sp, sp, 16
  ret
"""

SREG_NOT_SAVED = """\
.globl sreg_bad
sreg_bad:
  addi sp, sp, -16
  sw ra, 12(sp)
  li s1, 99
  lw ra, 12(sp)
  addi sp, sp, 16
  ret
"""

SREG_SAVED_NOT_RESTORED = """\
.globl sreg_half
sreg_half:
  addi sp, sp, -16
  sw ra, 12(sp)
  sw s2, 8(sp)
  addi s2, a0, 1
  lw ra, 12(sp)
  addi sp, sp, 16
  ret
"""

UNDEFINED_BRANCH_TARGET = """\
.globl branch_bad
branch_bad:
  addi sp, sp, -16
  sw ra, 12(sp)
  beq a0, zero, .nonexistent
  li a0, 0
.done:
  lw ra, 12(sp)
  addi sp, sp, 16
  ret
"""

GOOD_BRANCH = """\
.globl branch_ok
branch_ok:
  addi sp, sp, -16
  sw ra, 12(sp)
  beq a0, zero, .skip
  li a0, 1
.skip:
  lw ra, 12(sp)
  addi sp, sp, 16
  ret
"""

MULTIPLE_VIOLATIONS = """\
li s3, 10
beq a0, zero, .nowhere
addi sp, sp, -8
"""

NO_INSTRUCTIONS = """\
# just a comment
.section .text
"""


# ── Tests: Good function passes all rules ───────────────────────────────────

class TestGoodFunction:
    def test_passes_all_rules(self):
        result = lint_asm(GOOD_FUNC)
        assert result.passed is True
        assert result.violations == []

    def test_passes_specific_rule(self):
        for rule in ["func_label", "stack_balance", "ra_restore", "sreg_save", "label_refs"]:
            result = lint_asm(GOOD_FUNC, rules=[rule])
            assert result.passed is True, f"Rule {rule} unexpectedly failed"


# ── Tests: func_label rule ──────────────────────────────────────────────────

class TestFuncLabel:
    def test_missing_ret(self):
        result = lint_asm(MISSING_RET, rules=["func_label"])
        assert result.passed is False
        assert any(v.rule == "func_label" and "ret" in v.message.lower() for v in result.violations)

    def test_no_label(self):
        result = lint_asm("  add a0, a1, a2\n  ret\n", rules=["func_label"])
        assert result.passed is False
        assert any("label" in v.message.lower() for v in result.violations)

    def test_no_instructions(self):
        result = lint_asm(NO_INSTRUCTIONS, rules=["func_label"])
        assert result.passed is False

    def test_jr_ra_accepted(self):
        asm = "my_func:\n  li a0, 1\n  jr ra\n"
        result = lint_asm(asm, rules=["func_label"])
        assert result.passed is True


# ── Tests: stack_balance rule ───────────────────────────────────────────────

class TestStackBalance:
    def test_balanced(self):
        result = lint_asm(GOOD_FUNC, rules=["stack_balance"])
        assert result.passed is True

    def test_unbalanced(self):
        result = lint_asm(UNBALANCED_STACK, rules=["stack_balance"])
        assert result.passed is False
        v = result.violations[0]
        assert v.rule == "stack_balance"
        assert "-16" in v.message  # net delta = -32 + 16 = -16

    def test_no_stack_ops(self):
        asm = "my_func:\n  li a0, 1\n  ret\n"
        result = lint_asm(asm, rules=["stack_balance"])
        assert result.passed is True  # no stack ops = delta 0


# ── Tests: ra_restore rule ──────────────────────────────────────────────────

class TestRaRestore:
    def test_ra_properly_restored(self):
        result = lint_asm(GOOD_FUNC, rules=["ra_restore"])
        assert result.passed is True

    def test_ra_not_restored(self):
        result = lint_asm(MISSING_RA_RESTORE, rules=["ra_restore"])
        assert result.passed is False
        assert result.violations[0].rule == "ra_restore"

    def test_no_ra_save(self):
        asm = "my_func:\n  li a0, 1\n  ret\n"
        result = lint_asm(asm, rules=["ra_restore"])
        assert result.passed is True  # no save = nothing to check


# ── Tests: sreg_save rule ───────────────────────────────────────────────────

class TestSregSave:
    def test_sreg_properly_saved(self):
        result = lint_asm(GOOD_FUNC, rules=["sreg_save"])
        assert result.passed is True

    def test_sreg_not_saved(self):
        result = lint_asm(SREG_NOT_SAVED, rules=["sreg_save"])
        assert result.passed is False
        v = result.violations[0]
        assert v.rule == "sreg_save"
        assert "s1" in v.message

    def test_sreg_saved_not_restored(self):
        result = lint_asm(SREG_SAVED_NOT_RESTORED, rules=["sreg_save"])
        assert result.passed is False
        v = result.violations[0]
        assert "s2" in v.message
        assert "restored" in v.message.lower()


# ── Tests: label_refs rule ──────────────────────────────────────────────────

class TestLabelRefs:
    def test_valid_branch(self):
        result = lint_asm(GOOD_BRANCH, rules=["label_refs"])
        assert result.passed is True

    def test_undefined_target(self):
        result = lint_asm(UNDEFINED_BRANCH_TARGET, rules=["label_refs"])
        assert result.passed is False
        v = result.violations[0]
        assert v.rule == "label_refs"
        assert ".nonexistent" in v.message


# ── Tests: multiple violations ──────────────────────────────────────────────

class TestMultiViolations:
    def test_multiple_violations(self):
        result = lint_asm(MULTIPLE_VIOLATIONS)
        assert result.passed is False
        rules_hit = {v.rule for v in result.violations}
        # Should catch: no label, no ret, unbalanced stack, s3 not saved, undefined branch
        assert "func_label" in rules_hit
        assert "stack_balance" in rules_hit
        assert "sreg_save" in rules_hit
        assert "label_refs" in rules_hit


# ── Tests: API contract ────────────────────────────────────────────────────

class TestAPIContract:
    def test_to_dict(self):
        result = lint_asm(MISSING_RET, rules=["func_label"])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "passed" in d
        assert "violations" in d
        assert isinstance(d["violations"], list)
        assert all("rule" in v and "message" in v for v in d["violations"])

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown lint rules"):
            lint_asm("nop", rules=["bogus_rule"])

    def test_empty_input(self):
        result = lint_asm("")
        assert result.passed is False  # no label, no instructions
