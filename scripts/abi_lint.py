#!/usr/bin/env python3
"""
ABI Lint for RV32IM Assembly
============================
Static heuristic checks for common ABI / calling-convention issues
in generated RISC-V assembly text.

Library API
-----------
    from scripts.abi_lint import lint_asm, LintResult, Violation

    result: LintResult = lint_asm(asm_text, rules=None)
    # result.passed   -> bool
    # result.violations -> list[Violation]

CLI
---
    # Check a file (human-readable output):
    python -m scripts.abi_lint tests/fixtures/good_func.S

    # JSON output, specific rules only:
    python -m scripts.abi_lint input.S --rules func_label,stack_balance --json

    # Exit codes: 0 = pass, 1 = lint failure, 2 = usage error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Violation:
    """A single lint violation."""
    rule: str
    message: str
    line: Optional[int] = None          # 1-based line number, if applicable

@dataclass
class LintResult:
    """Aggregated lint result for one assembly text."""
    passed: bool
    violations: list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": [asdict(v) for v in self.violations],
        }


# ── Constants (defaults; may be overridden via configs/dataset.yaml) ─────────

ALL_RULES = [
    "func_label",
    "stack_balance",
    "ra_restore",
    "sreg_save",
    "label_refs",
]

CALLEE_SAVED = {f"s{i}" for i in range(12)}     # s0 – s11

BRANCH_MNEMONICS = {
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "j", "jal",
    "bnez", "beqz", "blez", "bgez", "bltz", "bgtz",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

_COMMENT_RE  = re.compile(r"#.*$")
_LABEL_RE    = re.compile(r"^([a-zA-Z_\.][a-zA-Z0-9_\.]*)\s*:")
_DIRECTIVE_RE = re.compile(r"^\s*\.")


def _strip_comment(line: str) -> str:
    """Remove trailing # comment from a line."""
    return _COMMENT_RE.sub("", line).strip()


def _parse_lines(text: str) -> list[str]:
    """Return list of stripped, comment-free lines (preserving index = line number - 1)."""
    return [_strip_comment(l) for l in text.splitlines()]


def _tokens(line: str) -> list[str]:
    """Very simple tokeniser: split on whitespace and commas."""
    return [t for t in re.split(r"[\s,]+", line) if t]


def _is_instruction(line: str) -> bool:
    """True if the line is (probably) an instruction, not a label/directive/blank."""
    if not line:
        return False
    if _LABEL_RE.match(line):
        return False
    if _DIRECTIVE_RE.match(line):
        return False
    return True


# ── Rule implementations ────────────────────────────────────────────────────

def _check_func_label(lines: list[str]) -> list[Violation]:
    """Rule: must contain at least one label and end with ret / jr ra."""
    violations: list[Violation] = []

    # 1. Look for at least one label
    has_label = False
    for i, ln in enumerate(lines):
        if _LABEL_RE.match(ln):
            has_label = True
            break
    if not has_label:
        violations.append(Violation("func_label", "No function label found", None))

    # 2. Last instruction must be ret or jr ra
    last_instr_idx: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if _is_instruction(lines[i]):
            last_instr_idx = i
            break

    if last_instr_idx is None:
        violations.append(Violation("func_label", "No instructions found", None))
    else:
        toks = _tokens(lines[last_instr_idx])
        is_ret = toks == ["ret"]
        is_jr_ra = toks[:2] == ["jr", "ra"] or toks[:2] == ["jalr", "ra"]
        if not (is_ret or is_jr_ra):
            violations.append(Violation(
                "func_label",
                f"Last instruction is '{lines[last_instr_idx]}', expected 'ret' or 'jr ra'",
                last_instr_idx + 1,
            ))

    return violations


def _check_stack_balance(lines: list[str]) -> list[Violation]:
    """Rule: addi sp, sp, -K must be matched by addi sp, sp, +K (net delta = 0)."""
    violations: list[Violation] = []
    sp_delta = 0  # cumulative

    sp_pattern = re.compile(
        r"addi\s+sp\s*,\s*sp\s*,\s*(-?\d+)", re.IGNORECASE
    )

    for i, ln in enumerate(lines):
        m = sp_pattern.search(ln)
        if m:
            sp_delta += int(m.group(1))

    if sp_delta != 0:
        violations.append(Violation(
            "stack_balance",
            f"Stack pointer delta is {sp_delta} (expected 0)",
            None,
        ))

    return violations


def _check_ra_restore(lines: list[str]) -> list[Violation]:
    """Rule: if ra is stored to stack, it must be reloaded before return."""
    violations: list[Violation] = []

    sw_ra = re.compile(r"sw\s+ra\s*,", re.IGNORECASE)
    lw_ra = re.compile(r"lw\s+ra\s*,", re.IGNORECASE)

    saved = False
    restored = False
    save_line: int | None = None

    for i, ln in enumerate(lines):
        if sw_ra.search(ln):
            saved = True
            save_line = i + 1
        if lw_ra.search(ln):
            restored = True

    if saved and not restored:
        violations.append(Violation(
            "ra_restore",
            "ra saved to stack but never restored before return",
            save_line,
        ))

    return violations


def _check_sreg_save(lines: list[str]) -> list[Violation]:
    """Rule: any callee-saved s-register written must be saved & restored."""
    violations: list[Violation] = []

    # Detect which s-regs are *written* (appear as first operand of ALU/load)
    written: dict[str, int] = {}   # reg -> first write line
    saved: set[str] = set()
    restored: set[str] = set()

    sw_re = re.compile(r"sw\s+(\w+)\s*,", re.IGNORECASE)
    lw_re = re.compile(r"lw\s+(\w+)\s*,", re.IGNORECASE)

    # Patterns where s-reg is the destination (first operand)
    dest_re = re.compile(
        r"(?:add|sub|sll|srl|sra|xor|or|and|slt|sltu|mul|mulh|mulhsu|mulhu|"
        r"div|divu|rem|remu|addi|slti|sltiu|xori|ori|andi|slli|srli|srai|"
        r"lb|lh|lw|lbu|lhu|lui|auipc|mv|li|la|not|neg|seqz|snez|sltz|sgtz)"
        r"\s+(s\d+)",
        re.IGNORECASE,
    )

    for i, ln in enumerate(lines):
        # Check for destination writes
        m = dest_re.search(ln)
        if m:
            reg = m.group(1).lower()
            if reg in CALLEE_SAVED and reg not in written:
                written[reg] = i + 1

        # Check for sw s-reg (save)
        m_sw = sw_re.search(ln)
        if m_sw:
            reg = m_sw.group(1).lower()
            if reg in CALLEE_SAVED:
                saved.add(reg)

        # Check for lw s-reg (restore)
        m_lw = lw_re.search(ln)
        if m_lw:
            reg = m_lw.group(1).lower()
            if reg in CALLEE_SAVED:
                restored.add(reg)

    for reg, line_no in sorted(written.items()):
        if reg not in saved:
            violations.append(Violation(
                "sreg_save",
                f"Callee-saved register {reg} written but not saved to stack",
                line_no,
            ))
        elif reg not in restored:
            violations.append(Violation(
                "sreg_save",
                f"Callee-saved register {reg} saved but not restored from stack",
                line_no,
            ))

    return violations


def _check_label_refs(lines: list[str]) -> list[Violation]:
    """Rule: all branch/jump label targets must be defined in the assembly."""
    violations: list[Violation] = []

    # Collect defined labels
    defined: set[str] = set()
    for ln in lines:
        m = _LABEL_RE.match(ln)
        if m:
            defined.add(m.group(1))

    # Scan for branch/jump instructions and extract target labels
    for i, ln in enumerate(lines):
        toks = _tokens(ln)
        if not toks:
            continue
        mnemonic = toks[0].lower()
        if mnemonic not in BRANCH_MNEMONICS:
            continue

        # Heuristic: last token that looks like a label (not a register, not a number)
        target = None
        for tok in reversed(toks[1:]):
            # skip registers (x0-x31, named regs) and immediates
            if re.match(r"^-?\d+$", tok):
                continue
            if re.match(r"^(x\d+|zero|ra|sp|gp|tp|t[0-6]|s\d+|a[0-7]|fp)$", tok, re.I):
                continue
            if re.match(r"^%", tok):  # relocations like %hi/%lo
                continue
            target = tok
            break

        if target and target not in defined:
            violations.append(Violation(
                "label_refs",
                f"Branch/jump target '{target}' is not defined",
                i + 1,
            ))

    return violations


# ── Dispatch table ───────────────────────────────────────────────────────────

_RULE_FNS = {
    "func_label":     _check_func_label,
    "stack_balance":  _check_stack_balance,
    "ra_restore":     _check_ra_restore,
    "sreg_save":      _check_sreg_save,
    "label_refs":     _check_label_refs,
}


# ── Public API ───────────────────────────────────────────────────────────────

def lint_asm(asm_text: str, rules: list[str] | None = None) -> LintResult:
    """
    Run ABI lint checks on an assembly text string.

    Parameters
    ----------
    asm_text : str
        The assembly source code to check.
    rules : list[str] | None
        List of rule IDs to run.  None = all rules.
        Valid IDs: func_label, stack_balance, ra_restore, sreg_save, label_refs

    Returns
    -------
    LintResult
        .passed = True if no violations, .violations = list[Violation]

    Input / Output Schema
    ---------------------
    Input:  plain-text assembly (str)
    Output: LintResult dataclass (convert via .to_dict() for JSON)
        {
            "passed": bool,
            "violations": [
                {"rule": str, "message": str, "line": int | null},
                ...
            ]
        }
    """
    if rules is None:
        rules = ALL_RULES

    unknown = set(rules) - set(ALL_RULES)
    if unknown:
        raise ValueError(f"Unknown lint rules: {unknown}. Valid: {ALL_RULES}")

    lines = _parse_lines(asm_text)
    all_violations: list[Violation] = []

    for rule_id in rules:
        fn = _RULE_FNS[rule_id]
        all_violations.extend(fn(lines))

    return LintResult(
        passed=len(all_violations) == 0,
        violations=all_violations,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="abi_lint",
        description="Static ABI/calling-convention lint for RV32IM assembly.",
        epilog="""
CLI usage examples:
  python -m scripts.abi_lint input.S
  python -m scripts.abi_lint input.S --rules func_label,stack_balance --json
  python -m scripts.abi_lint input.S --json | jq .violations
  cat generated.S | python -m scripts.abi_lint -       # read from stdin
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to .S file, or '-' for stdin",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help=f"Comma-separated rule IDs (default: all). Valid: {','.join(ALL_RULES)}",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    # Read input
    if args.input == "-":
        asm_text = sys.stdin.read()
    else:
        p = Path(args.input)
        if not p.exists():
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            sys.exit(2)
        asm_text = p.read_text()

    # Parse rules
    rules = None
    if args.rules:
        rules = [r.strip() for r in args.rules.split(",")]

    # Run lint
    try:
        result = lint_asm(asm_text, rules=rules)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Output
    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        if result.passed:
            print("PASS: No ABI lint violations found.")
        else:
            print(f"FAIL: {len(result.violations)} violation(s) found:\n")
            for v in result.violations:
                loc = f"line {v.line}" if v.line else "global"
                print(f"  [{v.rule}] ({loc}) {v.message}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
