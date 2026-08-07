#!/usr/bin/env python3
"""Teste de significância pareado para a avaliação do resolvedor MCKP.

Lê um benchmark_results.csv produzido por run_ollama_benchmark_matrix.py, separa
os casos em que o contexto excede o orçamento (binding) dos que já cabem e aplica,
para o par de estratégias escolhido, um teste t pareado e o teste de Wilcoxon
sinalizado. A variância provém dos casos distintos, não de repetições, o que é
apropriado quando a geração é determinística (temperatura 0).

Uso:
  .venv/bin/python scripts/paired_significance.py <benchmark_results.csv>
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

STRATS = ("mckp_uniform_control", "mckp", "raw")


def base_strategy(name: str) -> str:
    for b in STRATS:
        if b in name:
            return b
    return name


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load(path: str):
    cases: dict = defaultdict(dict)
    for row in csv.DictReader(open(path)):
        key = (row["benchmark"], row["test_case"])
        cases[key][base_strategy(row["strategy"])] = (
            as_float(row.get("score")),
            as_float(row.get("cost_original_tokens")),
            as_float(row.get("mckp_budget_tokens")),
        )
    return cases


def is_binding(entry: dict) -> bool:
    original = max((v[1] or 0) for v in entry.values())
    budget = next((v[2] for v in entry.values() if v[2]), None) or 1250
    return original > budget


def paired(cases: dict, a: str, b: str, keys: list) -> str:
    xs, ys = [], []
    for k in keys:
        d = cases[k]
        if a in d and b in d and d[a][0] is not None and d[b][0] is not None:
            xs.append(d[a][0])
            ys.append(d[b][0])
    diffs = [x - y for x, y in zip(xs, ys)]
    n = len(diffs)
    if n == 0:
        return f"{a} vs {b}: sem pares válidos"
    wins = sum(d > 0 for d in diffs)
    ties = sum(d == 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    out = (
        f"{a} vs {b}: n={n} meanDelta={st.mean(diffs):+.4f} "
        f"(vitorias {wins}, empates {ties}, derrotas {losses})"
    )
    if HAVE_SCIPY and n > 1 and any(d != 0 for d in diffs):
        _, p_t = stats.ttest_rel(xs, ys)
        try:
            _, p_w = stats.wilcoxon(xs, ys)
        except ValueError:
            p_w = float("nan")
        out += f" | t-test p={p_t:.4f} | Wilcoxon p={p_w:.4f}"
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("uso: paired_significance.py <benchmark_results.csv>", file=sys.stderr)
        raise SystemExit(2)
    cases = load(sys.argv[1])
    binding = [k for k, d in cases.items() if is_binding(d)]
    allk = list(cases)
    print(f"scipy disponivel: {HAVE_SCIPY}")
    print(f"casos totais={len(allk)} binding={len(binding)}\n")
    for label, keys in (("BINDING", binding), ("TODOS", allk)):
        print(f"=== {label} ===")
        print(paired(cases, "mckp", "mckp_uniform_control", keys))
        print(paired(cases, "mckp", "raw", keys))
        print()


if __name__ == "__main__":
    main()
