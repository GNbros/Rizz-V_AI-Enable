#!/usr/bin/env python3
"""
normalize_asm.py – Split and normalize a monolithic RISC-V assembly dump.

Input:  A single concatenated .S text file  (path from configs/dataset.yaml → paths.raw_input)
Output: One normalized .S file per snippet   (written to configs/dataset.yaml → paths.normalized_dir)

CLI usage:
    python scripts/normalize_asm.py --config configs/dataset.yaml
    python scripts/normalize_asm.py --config configs/dataset.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

import yaml


# ── Register alias tables ────────────────────────────────────────────────────

ABI_NAMES = {
    "x0": "zero", "x1": "ra", "x2": "sp", "x3": "gp", "x4": "tp",
    "x5": "t0", "x6": "t1", "x7": "t2",
    "x8": "s0", "x9": "s1",
    "x10": "a0", "x11": "a1", "x12": "a2", "x13": "a3",
    "x14": "a4", "x15": "a5", "x16": "a6", "x17": "a7",
    "x18": "s2", "x19": "s3", "x20": "s4", "x21": "s5",
    "x22": "s6", "x23": "s7", "x24": "s8", "x25": "s9",
    "x26": "s10", "x27": "s11",
    "x28": "t3", "x29": "t4", "x30": "t5", "x31": "t6",
}
XREG_NAMES = {v: k for k, v in ABI_NAMES.items()}
# fp is an alternate name for s0 / x8
ABI_NAMES_WITH_FP = {**ABI_NAMES}
XREG_NAMES_WITH_FP = {**XREG_NAMES, "fp": "x8"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def split_into_snippets(raw_text: str) -> List[Tuple[str, str]]:
    """Split a monolithic dump into (name, body) pairs.

    Heuristic: a new snippet starts at a line matching
        # <optional stars/dashes>
        # <name>.S
    or at a line that is just ``# See LICENSE for license details.`` when
    followed by a header block.
    """
    # Pattern: comment header containing a filename like "fcvt.S"
    header_re = re.compile(r"^#\s*(\w[\w\-]*\.S)\s*$")
    # Also detect "# See LICENSE..." as a snippet boundary when it marks
    # the beginning of a new concatenated file.
    license_re = re.compile(r"^#\s*See LICENSE for license details\.\s*$")

    lines = raw_text.splitlines(keepends=True)
    snippets: List[Tuple[str, List[str]]] = []
    current_name = "unknown"
    current_lines: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for license line that precedes a new header
        if license_re.match(line):
            # Look ahead to find a .S header within next 10 lines
            found_header = False
            for j in range(i + 1, min(i + 12, len(lines))):
                m = header_re.match(lines[j])
                if m:
                    # Save current snippet
                    if current_lines:
                        snippets.append((current_name, current_lines))
                    current_name = m.group(1).replace(".S", "")
                    current_lines = []
                    i = j + 1
                    found_header = True
                    break
            if found_header:
                continue

        # Check for inline .S header (files without license preamble)
        m = header_re.match(line)
        if m and current_lines:
            snippets.append((current_name, current_lines))
            current_name = m.group(1).replace(".S", "")
            current_lines = []
            i += 1
            continue

        current_lines.append(line)
        i += 1

    # Flush last snippet
    if current_lines:
        snippets.append((current_name, current_lines))

    return [(name, "".join(body)) for name, body in snippets]


def strip_line_comments(text: str) -> str:
    """Remove # comments but keep preprocessor directives (#include, #define, #if, etc.)."""
    out_lines = []
    preprocessor_re = re.compile(
        r"^\s*#\s*(include|define|undef|if|ifdef|ifndef|elif|else|endif|pragma|error|warning)\b"
    )
    # Comment-only line: starts with # but is not a preprocessor directive
    comment_line_re = re.compile(r"^\s*#")
    # Inline comment after code: match ' # comment' but not inside strings
    inline_comment_re = re.compile(r"^(.*?\S)\s+#\s.*$")

    for line in text.splitlines():
        # Keep preprocessor lines as-is
        if preprocessor_re.match(line):
            out_lines.append(line)
            continue
        # Pure comment line → skip
        if comment_line_re.match(line):
            continue
        # Strip inline comment
        m = inline_comment_re.match(line)
        if m:
            out_lines.append(m.group(1))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def strip_block_comments(text: str) -> str:
    """Remove /* ... */ block comments."""
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines to at most one."""
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def normalize_indent(text: str, spaces: int) -> str:
    """Re-indent lines that use leading whitespace (not labels/directives at col 0)."""
    if spaces <= 0:
        return text
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped:
            out.append("")
        elif line[0] in (" ", "\t"):
            out.append(" " * spaces + stripped)
        else:
            out.append(stripped)
    return "\n".join(out) + "\n"


def apply_register_aliases(text: str, direction: str) -> str:
    """Replace register names according to direction ('abi' or 'xreg').

    'abi'  → canonical ABI names  (x0 → zero, x1 → ra, …)
    'xreg' → canonical x-register names (zero → x0, ra → x1, …)
    """
    if direction == "abi":
        table = ABI_NAMES_WITH_FP
    elif direction == "xreg":
        table = XREG_NAMES_WITH_FP
    else:
        return text  # no aliasing

    # Sort by length descending so x10 is matched before x1
    sorted_keys = sorted(table.keys(), key=len, reverse=True)
    # Build one big regex: word-boundary match
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b")
    return pattern.sub(lambda m: table[m.group(0)], text)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split and normalize a monolithic RISC-V .S dump.",
        epilog=textwrap.dedent("""\
            examples:
              python scripts/normalize_asm.py --config configs/dataset.yaml
              python scripts/normalize_asm.py --config configs/dataset.yaml --dry-run
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to dataset.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(args.config).resolve().parent.parent
    raw_path = project_root / cfg["paths"]["raw_input"]
    out_dir = project_root / cfg["paths"]["normalized_dir"]
    norm_cfg = cfg.get("normalize", {})

    if not raw_path.exists():
        print(f"ERROR: raw input not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    snippets = split_into_snippets(raw_text)
    print(f"[normalize] Split into {len(snippets)} snippets from {raw_path.name}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx, (name, body) in enumerate(snippets, start=1):
        text = body

        # 1. Block comments
        if norm_cfg.get("strip_block_comments", True):
            text = strip_block_comments(text)

        # 2. Line comments
        if norm_cfg.get("strip_comments", True):
            text = strip_line_comments(text)

        # 3. Blank lines
        if norm_cfg.get("collapse_blank_lines", True):
            text = collapse_blank_lines(text)

        # 4. Indent
        indent = norm_cfg.get("indent_spaces", 0)
        if indent > 0:
            text = normalize_indent(text, indent)

        # 5. Register aliases
        direction = norm_cfg.get("register_alias_direction", "")
        if direction:
            text = apply_register_aliases(text, direction)

        # Skip empty results
        if not text.strip():
            continue

        fname = f"{idx:03d}_{name}.S"
        if args.dry_run:
            print(f"  [dry-run] {fname}  ({len(text.splitlines())} lines)")
        else:
            (out_dir / fname).write_text(text, encoding="utf-8")
            written += 1

    if args.dry_run:
        print(f"[dry-run] Would write {len(snippets)} files to {out_dir}")
    else:
        print(f"[normalize] Wrote {written} files to {out_dir}")


if __name__ == "__main__":
    main()
