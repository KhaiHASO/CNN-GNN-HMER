from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatexValidationResult:
    ok: bool
    error: str | None = None


def clean_latex(latex_raw: str) -> str:
    latex = " ".join((latex_raw or "").strip().split())
    if latex.startswith("$") and latex.endswith("$") and len(latex) > 1:
        latex = latex[1:-1].strip()
    if latex.startswith("\\(") and latex.endswith("\\)"):
        latex = latex[2:-2].strip()
    if latex.startswith("\\[") and latex.endswith("\\]"):
        latex = latex[2:-2].strip()
    return latex


def validate_latex_basic(latex: str) -> LatexValidationResult:
    if not latex.strip():
        return LatexValidationResult(False, "LaTeX rỗng")
    stack: list[str] = []
    pairs = {"}": "{", ")": "(", "]": "["}
    escaped = False
    for char in latex:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in "{([":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return LatexValidationResult(False, "Ngoặc không cân bằng")
    if stack:
        return LatexValidationResult(False, "Ngoặc không cân bằng")
    if "\\begin" in latex and "\\end" not in latex:
        return LatexValidationResult(False, "Thiếu \\end")
    return LatexValidationResult(True)

