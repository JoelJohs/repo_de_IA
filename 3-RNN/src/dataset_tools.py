"""Tools to clean, split, and count C functions for the dataset."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


THEME_KEYWORDS = {
    "arrays": ["array", "matrix", "vector"],
    "trees": ["tree", "avl", "bst", "redblack", "rb", "heap"],
    "lists": ["list", "linked", "stack", "queue", "node"],
    "sorting": ["sort", "merge", "quick", "bubble", "insert", "selection"],
    "math": ["math", "sqrt", "sin", "cos", "tan", "pow", "log", "fibo", "prime", "gcd", "lcm"],
    "strings": ["string", "strcmp", "strlen", "strcpy", "strncpy", "strcat", "memcmp"],
    "io_system": ["stdio", "printf", "scanf", "file", "open", "read", "write", "socket", "thread", "pthread", "fork", "exec", "signal", "pipe", "time", "clock"],
}

CONTROL_KEYWORDS = {"if", "for", "while", "switch", "case", "return"}


@dataclass
class FunctionEntry:
    source: str
    path: str
    code: str


def normalize_code(code: str) -> str:
    return re.sub(r"\s+", "", code)


def detect_theme(code: str, path: str) -> str:
    haystack = f"{path}\n{code}".lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return theme
    return "misc"


def read_jsonl_functions(jsonl_path: Path) -> Iterable[FunctionEntry]:
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", {})
            func_def = text.get("func_def")
            if not func_def:
                continue
            path = text.get("path", str(jsonl_path))
            yield FunctionEntry("jsonl", path, func_def.strip())


def looks_like_function_signature(signature: str) -> bool:
    signature = signature.strip()
    if not signature or "(" not in signature or ")" not in signature:
        return False
    if signature.startswith("#"):
        return False
    first_word = signature.split()[0]
    if first_word in CONTROL_KEYWORDS:
        return False
    matches = list(re.finditer(r"([A-Za-z_]\w*)\s*\(", signature))
    if not matches:
        return False
    name = matches[-1].group(1)
    if name in CONTROL_KEYWORDS:
        return False
    return True


def extract_functions_from_c(text: str, source_path: str) -> list[FunctionEntry]:
    lines = text.splitlines()
    results: list[FunctionEntry] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "{" not in line:
            i += 1
            continue
        signature_lines = [line]
        sig_start = i
        while "{" not in " ".join(signature_lines) and sig_start > 0:
            sig_start -= 1
            signature_lines.insert(0, lines[sig_start])
            if len(signature_lines) > 6:
                break
        signature = " ".join(signature_lines)
        if not looks_like_function_signature(signature):
            i += 1
            continue
        brace_count = line.count("{") - line.count("}")
        j = i + 1
        while j < len(lines) and brace_count > 0:
            brace_count += lines[j].count("{") - lines[j].count("}")
            j += 1
        if brace_count != 0:
            i += 1
            continue
        func_text = "\n".join(lines[sig_start:j]).strip()
        results.append(FunctionEntry("c", source_path, func_text))
        i = j
    return results


def write_theme_files(out_dir: Path, theme_map: dict[str, list[FunctionEntry]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme, items in sorted(theme_map.items()):
        if not items:
            continue
        content = "\n\n".join(entry.code for entry in items).strip() + "\n"
        (out_dir / f"{theme}.c").write_text(content, encoding="utf-8")


def write_report(out_dir: Path, totals: dict[str, int], total_all: int) -> None:
    lines = [
        "# Reporte de dataset",
        "",
        f"Total de funciones: {total_all}",
        "",
        "## Por tema",
    ]
    for theme, count in sorted(totals.items()):
        lines.append(f"- {theme}: {count}")
    lines.append("")
    (out_dir / "REPORTE.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and split C dataset.")
    parser.add_argument(
        "--c-input",
        type=Path,
        default=Path("dataset/funciones.c"),
        help="Path to C file with raw functions.",
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory with jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset/clean"),
        help="Output directory for cleaned files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entries: list[FunctionEntry] = []

    if args.c_input.exists():
        text = args.c_input.read_text(encoding="utf-8", errors="ignore")
        entries.extend(extract_functions_from_c(text, str(args.c_input)))

    if args.jsonl_dir.exists():
        for jsonl_path in sorted(args.jsonl_dir.glob("*.jsonl")):
            entries.extend(read_jsonl_functions(jsonl_path))

    deduped: list[FunctionEntry] = []
    seen: set[str] = set()
    for entry in entries:
        key = normalize_code(entry.code)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    theme_map: dict[str, list[FunctionEntry]] = {"misc": []}
    for theme in THEME_KEYWORDS:
        theme_map.setdefault(theme, [])

    for entry in deduped:
        theme = detect_theme(entry.code, entry.path)
        theme_map.setdefault(theme, []).append(entry)

    write_theme_files(args.out_dir, theme_map)

    totals = {theme: len(items) for theme, items in theme_map.items()}
    total_all = sum(totals.values())
    write_report(args.out_dir, totals, total_all)

    print(f"Total functions: {total_all}")
    for theme, count in sorted(totals.items()):
        print(f"{theme}: {count}")


if __name__ == "__main__":
    main()
