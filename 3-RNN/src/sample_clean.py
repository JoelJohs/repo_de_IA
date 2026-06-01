"""Sample 'clean' C functions from a large corpus file.

Reads `dataset/funciones.c` (or any big C file) and extracts N functions
that match style heuristics required by the activity:

  * pure ASCII (no comments in other languages)
  * reasonable length (30 - 1500 chars)
  * no main() (we want libraries, not entry points)
  * single, recognizable function signature
  * balanced braces
  * no exotic preprocessor or system includes

Output: `dataset/sample/funciones.c` + `dataset/sample/REPORTE.md`
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

CONTROL_KEYWORDS = {"if", "for", "while", "switch", "case", "return", "do", "else"}
SIG_RE = re.compile(
    r"^\s*(?:static\s+)?(?:inline\s+)?"
    r"(?:int|void|char|float|double|long|short|unsigned|signed|size_t|ssize_t|bool|"
    r"int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t)\s*\*?\s+"
    r"([A-Za-z_]\w*)\s*\([^;]*\)\s*\{",
    re.MULTILINE,
)


def is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def balanced_braces(text: str) -> bool:
    opens = text.count("{")
    closes = text.count("}")
    return opens == closes and opens > 0


def has_main(text: str) -> bool:
    return re.search(r"\bint\s+main\s*\(", text) is not None


def has_exotic_preproc(text: str) -> bool:
    return bool(re.search(r"^\s*#\s*(include|define|ifndef|ifdef|endif|pragma)", text, re.MULTILINE))


def has_chinese_comment(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def count_style(text: str) -> dict[str, int]:
    return {
        "lines": text.count("\n"),
        "braces": text.count("{"),
        "parens": text.count("("),
        "loops": len(re.findall(r"\b(for|while)\s*\(", text)),
        "ifs": len(re.findall(r"\bif\s*\(", text)),
        "returns": len(re.findall(r"\breturn\b", text)),
    }


def is_clean_function(code: str) -> tuple[bool, str]:
    """Return (ok, reason_if_rejected)."""
    n = len(code)
    if n < 30 or n > 1500:
        return False, f"length={n}"
    if has_main(code):
        return False, "has_main"
    if has_chinese_comment(code):
        return False, "non_ascii"
    if not is_ascii(code):
        return False, "non_ascii"
    if not balanced_braces(code):
        return False, "unbalanced_braces"
    m = SIG_RE.search(code)
    if not m:
        return False, "no_signature"
    if m.group(1) in CONTROL_KEYWORDS:
        return False, "control_keyword_name"
    return True, "ok"


def extract_functions(text: str) -> Iterable[tuple[str, str]]:
    """Yield (signature_line, full_block) for every detected function."""
    matches = list(SIG_RE.finditer(text))
    for i, m in enumerate(matches):
        sig_start = m.start()
        # find the matching close brace by counting
        depth = 0
        j = m.end() - 1  # position of the '{' character
        for j in range(j, len(text)):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
        else:
            continue  # unbalanced, skip
        block = text[sig_start : j + 1]
        yield m.group(0).strip().split("\n")[0].strip(), block


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample clean C functions from a large corpus.")
    p.add_argument("--input", type=Path, default=Path("dataset/funciones.c"))
    p.add_argument("--out-dir", type=Path, default=Path("dataset/sample"))
    p.add_argument("--target", type=int, default=150, help="Number of clean functions to keep.")
    p.add_argument(
        "--max-bytes",
        type=int,
        default=20_000_000,
        help="Read at most this many bytes from the input (functions per MB ~ 2000).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading first {args.max_bytes:,} bytes of {args.input} ...")
    with args.input.open("rb") as fh:
        raw = fh.read(args.max_bytes)
    text = raw.decode("utf-8", errors="ignore")
    print(f"Loaded {len(text):,} chars.")

    candidates: list[tuple[str, str]] = []
    rejected: Counter[str] = Counter()

    # find all function starts in one pass (O(n) regex)
    starts = [m.start() for m in SIG_RE.finditer(text)]
    print(f"Found {len(starts):,} candidate signatures.")

    # for each start, find the matching closing brace by depth tracking
    text_chars = text
    next_start = (starts[1:] + [len(text_chars)]) if starts else []
    for i, sig_start in enumerate(starts):
        cap = next_start[i]
        # scan from sig_start forward
        depth = 0
        seen_open = False
        end = cap  # fallback
        for j in range(sig_start, cap):
            c = text_chars[j]
            if c == "{":
                depth += 1
                seen_open = True
            elif c == "}":
                depth -= 1
                if seen_open and depth == 0:
                    end = j + 1
                    break
        block = text_chars[sig_start:end]
        sig_line = text_chars[sig_start : sig_start + 200].split("\n", 1)[0].strip()
        ok, reason = is_clean_function(block)
        if not ok:
            rejected[reason] += 1
            continue
        candidates.append((sig_line, block))

    print(f"Scanned, kept {len(candidates)} candidates. Rejected: {dict(rejected)}")
    if not candidates:
        raise SystemExit("No clean candidates found. Loosen heuristics in is_clean_function().")

    # score: prefer functions with control flow and a clean signature
    def score(item: tuple[str, str]) -> int:
        s = count_style(item[1])
        return s["returns"] * 2 + s["ifs"] + s["loops"] - abs(200 - len(item[1])) // 50

    candidates.sort(key=score, reverse=True)
    chosen = candidates[: args.target]

    out = args.out_dir / "funciones.c"
    parts: list[str] = []
    for sig, block in chosen:
        parts.append(block.strip())
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {len(chosen)} functions to {out} ({out.stat().st_size:,} bytes)")

    # style counts report
    style_aggregate: Counter[str] = Counter()
    for _, block in chosen:
        style_aggregate.update(count_style(block))

    rep = args.out_dir / "REPORTE.md"
    lines = [
        "# Reporte de muestreo",
        "",
        f"Input: `{args.input}`",
        f"Funciones seleccionadas: {len(chosen)}",
        f"Rechazadas: {sum(rejected.values())}",
        "",
        "## Razones de rechazo",
        "",
    ]
    for reason, n in rejected.most_common():
        lines.append(f"- `{reason}`: {n}")
    lines += [
        "",
        "## Estilo agregado del muestreo",
        "",
        f"- Líneas totales: {style_aggregate['lines']}",
        f"- Llaves abiertas: {style_aggregate['braces']}",
        f"- Loops (for/while): {style_aggregate['loops']}",
        f"- Condicionales (if): {style_aggregate['ifs']}",
        f"- Returns: {style_aggregate['returns']}",
        "",
        "## Muestra (5 funciones al azar)",
        "",
    ]
    import random

    random.seed(42)
    for _, block in random.sample(chosen, k=min(5, len(chosen))):
        lines.append("```c")
        lines.append(block.strip())
        lines.append("```")
        lines.append("")
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"Reporte en {rep}")


if __name__ == "__main__":
    main()
