"""Curate the best C functions per category from dataset/clean/.

For each `dataset/clean/<category>.c`, this script:

  1. Extracts every candidate function (signature + balanced braces).
  2. Applies strict filters (ASCII, length, control flow, no main/extern/pragma).
  3. Scores survivors on style and pedagogical value.
  4. Keeps the top 50-80 functions per category.

Output (in --out-dir):

  * `<category>.c`         - the chosen functions, blank-line separated
  * `funciones.c`          - all categories concatenated (input to preprocess.py)
  * `REPORTE.md`           - counts, rejection reasons, top scores per category

This replaces the random sampling done by `sample_clean.py` (which works on the
un-categorized 466 MB corpus). The output here is a smaller, higher-signal
training set that exercises the same eight themes the model will see at
inference time.
"""

from __future__ import annotations

import argparse
import heapq
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
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

CATEGORIES = ("arrays", "io_system", "lists", "math", "misc", "sorting", "strings", "trees")


@dataclass
class Candidate:
    sig: str
    body: str
    score: int
    n_lines: int
    n_chars: int


def is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def has_main(text: str) -> bool:
    return re.search(r"\bint\s+main\s*\(", text) is not None


def has_chinese(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def is_balanced(text: str) -> bool:
    return text.count("{") == text.count("}") and "{" in text


def bad_patterns(text: str) -> str | None:
    """Return a short reason if the function contains 'noisy' patterns we want
    to penalize. None means OK."""
    if re.search(r"^\s*#\s*(include|define|ifndef|ifdef|endif|pragma)", text, re.MULTILINE):
        return "preprocessor"
    if re.search(r"\bextern\s+\w", text):
        return "extern_global"
    if re.search(r"^\s*static\s+\w+\s+\w+\s*=\s*", text, re.MULTILINE) and text.count(";") < 3:
        return "single_decl"
    if text.count("//") > 5 or text.count("/*") > 2:
        return "comment_heavy"
    return None


def count_style(text: str) -> dict[str, int]:
    return {
        "lines": text.count("\n"),
        "braces": text.count("{"),
        "parens": text.count("("),
        "loops": len(re.findall(r"\b(for|while)\s*\(", text)),
        "ifs": len(re.findall(r"\bif\s*\(", text)),
        "returns": len(re.findall(r"\breturn\b", text)),
    }


def detect_indent(text: str) -> int:
    """Return 2, 4, or 0 if no consistent indentation found."""
    indents = []
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("\t"):
            return -1
        n = len(line) - len(stripped)
        if 0 < n <= 8:
            indents.append(n)
    if not indents:
        return 0
    most = Counter(indents).most_common(1)[0][0]
    if most in (2, 4) and indents.count(most) / len(indents) > 0.6:
        return most
    return 0


def score_function(body: str) -> int:
    s = count_style(body)
    if s["returns"] == 0:
        return -1000
    if s["loops"] + s["ifs"] == 0:
        return -500
    n = len(body)
    score = (
        s["returns"] * 3
        + s["ifs"] * 2
        + s["loops"] * 2
        - abs(400 - n) // 50
        - abs(12 - s["lines"]) // 4
    )
    ind = detect_indent(body)
    if ind in (2, 4):
        score += 4
    if bad_patterns(body) is not None:
        score -= 8
    if s["parens"] > 60:
        score -= 4
    if s["lines"] < 4 or s["lines"] > 60:
        score -= 6
    return score


def extract_function_blocks(text: str) -> Iterable[tuple[str, str]]:
    """Yield (signature_line, body) for every recognized function.

    Uses the same algorithm as sample_clean.py: SIG_RE finds candidates, then
    we walk forward counting braces to find the matching close.
    """
    starts = [m.start() for m in SIG_RE.finditer(text)]
    if not starts:
        return
    next_starts = starts[1:] + [len(text)]
    for raw_start, cap in zip(starts, next_starts):
        m = SIG_RE.match(text, raw_start)
        if not m:
            continue
        name = m.group(1)
        if name in CONTROL_KEYWORDS:
            continue
        # The regex uses ^\s* which in MULTILINE eats the previous \n.
        # Skip past the leading whitespace to point at the real signature.
        sig_start = raw_start
        while sig_start < cap and text[sig_start] in " \t\r\n":
            sig_start += 1
        if sig_start >= cap:
            continue
        sig_line = text[sig_start : sig_start + 200].split("\n", 1)[0].strip()
        depth = 0
        seen_open = False
        end = cap
        for j in range(sig_start, cap):
            c = text[j]
            if c == "{":
                depth += 1
                seen_open = True
            elif c == "}":
                depth -= 1
                if seen_open and depth == 0:
                    end = j + 1
                    break
        body = text[sig_start:end]
        if not is_balanced(body):
            continue
        yield sig_line, body


def filter_candidates(text: str, args: argparse.Namespace) -> tuple[list[Candidate], Counter[str], int]:
    """Return (kept_candidates, rejected_reasons, total_scanned)."""
    kept: list[Candidate] = []
    rejected: Counter[str] = Counter()
    scanned = 0

    for sig_line, body in extract_function_blocks(text):
        scanned += 1
        n = len(body)
        if n < args.length_min or n > args.length_max:
            rejected["length"] += 1
            continue
        if has_main(body):
            rejected["main"] += 1
            continue
        if has_chinese(body):
            rejected["non_ascii"] += 1
            continue
        if not is_ascii(body):
            rejected["non_ascii"] += 1
            continue
        if not is_balanced(body):
            rejected["unbalanced"] += 1
            continue
        style = count_style(body)
        if style["returns"] < args.min_returns:
            rejected["no_return"] += 1
            continue
        if style["loops"] + style["ifs"] < args.min_control:
            rejected["no_control_flow"] += 1
            continue
        reason = bad_patterns(body)
        if reason == "preprocessor" or reason == "extern_global":
            rejected[reason] += 1
            continue
        sc = score_function(body)
        if sc < 0:
            rejected["negative_score"] += 1
            continue
        kept.append(
            Candidate(
                sig=sig_line,
                body=body.strip(),
                score=sc,
                n_lines=style["lines"],
                n_chars=n,
            )
        )

    kept.sort(key=lambda c: c.score, reverse=True)
    return kept, rejected, scanned


def select_top(kept: list[Candidate], per_cat_min: int, per_cat_max: int) -> tuple[list[Candidate], str | None]:
    """Pick the top N candidates. If fewer than `min` survive, return them all
    and a warning. If more than `max` survive, take the top `max`."""
    if len(kept) < per_cat_min:
        return kept, f"solo {len(kept)} (objetivo {per_cat_min})"
    if len(kept) > per_cat_max:
        return kept[:per_cat_max], None
    return kept, None


def curate_one(theme: str, src: Path, args: argparse.Namespace) -> tuple[list[Candidate], Counter[str], int, str | None]:
    print(f"  [{theme}] leyendo {src} ({src.stat().st_size:,} bytes)...")
    t0 = time.time()
    text = src.read_text(encoding="utf-8", errors="ignore")
    t1 = time.time()
    print(f"  [{theme}] cargado en {t1 - t0:.1f}s ({len(text):,} chars)")
    kept, rejected, scanned = filter_candidates(text, args)
    chosen, warn = select_top(kept, args.per_cat_min, args.per_cat_max)
    print(
        f"  [{theme}] escaneadas={scanned:,}  válidas={len(kept):,}  "
        f"elegidas={len(chosen)}  rechazos={dict(rejected)}  "
        f"({time.time() - t1:.1f}s)"
    )
    return chosen, rejected, scanned, warn


def write_per_category(out_dir: Path, theme: str, chosen: list[Candidate]) -> Path:
    out = out_dir / f"{theme}.c"
    content = "\n\n".join(c.body for c in chosen).strip() + "\n"
    out.write_text(content, encoding="utf-8")
    return out


def write_combined(out_dir: Path, all_chosen: dict[str, list[Candidate]]) -> Path:
    parts: list[str] = []
    for theme in CATEGORIES:
        for c in all_chosen.get(theme, []):
            parts.append(c.body)
    out = out_dir / "funciones.c"
    out.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")
    return out


def write_report(
    out_dir: Path,
    all_chosen: dict[str, list[Candidate]],
    all_rejected: dict[str, Counter[str]],
    all_scanned: dict[str, int],
    warnings: dict[str, str | None],
    args: argparse.Namespace,
) -> Path:
    total_kept = sum(len(c) for c in all_chosen.values())
    total_scanned = sum(all_scanned.values())
    lines: list[str] = [
        "# Reporte de curación por categoría",
        "",
        f"Input dir:  `{args.clean_dir}`",
        f"Output dir: `{out_dir}`",
        f"Heurística: length={args.length_min}-{args.length_max}, "
        f"min_returns={args.min_returns}, min_control={args.min_control}",
        f"Objetivo por categoría: {args.per_cat_min}-{args.per_cat_max}",
        "",
        "## Resumen",
        "",
        f"- Funciones escaneadas: {total_scanned:,}",
        f"- Funciones seleccionadas: {total_kept}",
        "",
        "## Por categoría",
        "",
    ]
    for theme in CATEGORIES:
        n = len(all_chosen.get(theme, []))
        warn = warnings.get(theme)
        note = f" — ⚠ {warn}" if warn else ""
        lines.append(f"- **{theme}**: {n} funciones{note}")

    lines += ["", "## Razones de rechazo agregadas", ""]
    agg: Counter[str] = Counter()
    for c in all_rejected.values():
        agg.update(c)
    for reason, n in agg.most_common():
        lines.append(f"- `{reason}`: {n:,}")
    lines.append("")

    lines += ["## Top-3 funciones (por score) por categoría", ""]
    for theme in CATEGORIES:
        chosen = all_chosen.get(theme, [])
        if not chosen:
            lines.append(f"### {theme}")
            lines.append("")
            lines.append("_(sin funciones elegidas)_")
            lines.append("")
            continue
        lines.append(f"### {theme}")
        lines.append("")
        for c in chosen[:3]:
            sig = c.sig[:100]
            lines.append(f"- score={c.score} | {c.n_lines} líneas | {c.n_chars} chars | `{sig}`")
        lines.append("")

    out = out_dir / "REPORTE.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curate the best 50-80 C functions per category from dataset/clean/."
    )
    p.add_argument("--clean-dir", type=Path, default=Path("dataset/clean"))
    p.add_argument("--out-dir", type=Path, default=Path("dataset/curated"))
    p.add_argument("--per-cat-min", type=int, default=50)
    p.add_argument("--per-cat-max", type=int, default=80)
    p.add_argument("--length-min", type=int, default=200)
    p.add_argument("--length-max", type=int, default=600)
    p.add_argument("--min-returns", type=int, default=1)
    p.add_argument("--min-control", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.clean_dir.is_dir():
        print(f"No se encontró {args.clean_dir}", file=sys.stderr)
        sys.exit(1)

    all_chosen: dict[str, list[Candidate]] = {}
    all_rejected: dict[str, Counter[str]] = {}
    all_scanned: dict[str, int] = {}
    warnings: dict[str, str | None] = {}

    for theme in CATEGORIES:
        src = args.clean_dir / f"{theme}.c"
        if not src.is_file():
            print(f"  [{theme}] falta {src}, saltando", file=sys.stderr)
            warnings[theme] = "archivo no encontrado"
            all_chosen[theme] = []
            all_rejected[theme] = Counter()
            all_scanned[theme] = 0
            continue
        chosen, rejected, scanned, warn = curate_one(theme, src, args)
        all_chosen[theme] = chosen
        all_rejected[theme] = rejected
        all_scanned[theme] = scanned
        warnings[theme] = warn
        write_per_category(args.out_dir, theme, chosen)

    combined = write_combined(args.out_dir, all_chosen)
    rep = write_report(args.out_dir, all_chosen, all_rejected, all_scanned, warnings, args)

    total = sum(len(c) for c in all_chosen.values())
    print()
    print(f"Total elegidas: {total} funciones en {len(CATEGORIES)} categorías")
    print(f"Combinado:    {combined} ({combined.stat().st_size:,} bytes)")
    print(f"Reporte:      {rep}")


if __name__ == "__main__":
    main()
