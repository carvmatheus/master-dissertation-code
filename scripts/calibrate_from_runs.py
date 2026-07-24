#!/usr/bin/env python3
"""
Agregador de calibração: destila os CSVs da matriz de benchmarks nos parâmetros
L (custo) e Q (qualidade) da formulação knapsack, por (estratégia, benchmark,
num_ctx), e mede se os validadores predizem o acerto downstream.

Entrada:  <output-root>/runs/<model>/ctx-<N>/benchmark_results.csv
Saída:
  - calibration_LQ.csv       -> L/Q médios por (estrategia, benchmark, ctx)
  - calibration_corr.csv     -> correlação de cada métrica de qualidade com o score
  - imprime um resumo legível no stdout

Uso:
  python scripts/calibrate_from_runs.py [--output-root DIR] [--out DIR]

Sem dependências além da stdlib — roda mesmo com a matriz ainda em andamento
(agrega o que já existe).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# Métricas de qualidade (Q) e custo (L) escritas pelos validadores.
QUALITY_COLS = [
    "quality_content_recall",
    "quality_number_retention",
    "quality_evidence_recall",
    "quality_answer_present",
    "quality_semantic_fidelity",
    "quality_info_density",
]
COST_COLS = [
    "cost_compression_ratio",
    "cost_tokens_saved",
    "cost_compressed_tokens",
    "cost_original_tokens",
    "cost_char_ratio",
]


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _strategy_base(name: str) -> str:
    """raw_llama3.1-8b-instruct-q8 -> raw (remove o sufixo do modelo)."""
    for base in ("raw", "sliding_window", "parallel_window",
                 "semantic_compression", "rig", "rsaw", "mock"):
        if name.startswith(base):
            return base
    return name


def load_rows(output_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    runs_dir = output_root / "runs"
    if not runs_dir.exists():
        return rows
    for csv_path in sorted(runs_dir.glob("*/ctx-*/benchmark_results.csv")):
        m = re.search(r"ctx-(\d+)", str(csv_path.parent.name))
        num_ctx = int(m.group(1)) if m else -1
        model = csv_path.parent.parent.name
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                r["_num_ctx"] = num_ctx
                r["_model"] = model
                r["_strategy_base"] = _strategy_base(r.get("strategy", ""))
                rows.append(r)
    return rows


def aggregate_lq(rows: List[Dict]) -> List[Dict]:
    """L/Q médios por (estrategia, benchmark, num_ctx)."""
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r["_strategy_base"], r.get("benchmark", ""), r["_num_ctx"])
        groups[key].append(r)

    out: List[Dict] = []
    for (strat, bench, ctx), grp in sorted(groups.items()):
        rec: Dict[str, object] = {
            "strategy": strat,
            "benchmark": bench,
            "num_ctx": ctx,
            "n": len(grp),
            "avg_score": _mean([_to_float(r.get("score")) for r in grp]),
            "avg_latency_ms": _mean([_to_float(r.get("latency_ms")) for r in grp]),
        }
        for col in COST_COLS + QUALITY_COLS:
            rec[col] = _mean([_to_float(r.get(col)) for r in grp])
        out.append(rec)
    return out


def correlations(rows: List[Dict]) -> List[Dict]:
    """Correlação de Pearson de cada métrica de qualidade/custo com o score,
    considerando só linhas onde houve compressão de verdade (ratio < 1)."""
    scored = [r for r in rows if _to_float(r.get("score")) is not None]
    compressed = [
        r for r in scored
        if (_to_float(r.get("cost_compression_ratio")) or 1.0) < 0.999
    ]
    out: List[Dict] = []
    for scope_name, subset in (("all", scored), ("compressed_only", compressed)):
        scores = [_to_float(r.get("score")) for r in subset]
        for col in QUALITY_COLS + ["cost_compression_ratio", "cost_tokens_saved"]:
            metric = [_to_float(r.get(col)) for r in subset]
            out.append({
                "scope": scope_name,
                "metric": col,
                "pearson_vs_score": _pearson(metric, scores),
                "n": sum(1 for a, b in zip(metric, scores)
                         if a is not None and b is not None),
            })
    return out


def _fmt(v: object, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_csv(path: Path, records: List[Dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for rec in records:
            w.writerow({c: _fmt(rec.get(c)) if isinstance(rec.get(c), float)
                        else rec.get(c) for c in columns})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path,
                    default=REPO_ROOT / "tests" / "ollama-local" / "benchmark_calibration_ampla")
    ap.add_argument("--out", type=Path, default=None,
                    help="Diretório de saída (padrão: <output-root>/calibration)")
    args = ap.parse_args()
    out_dir = args.out or (args.output_root / "calibration")

    rows = load_rows(args.output_root)
    if not rows:
        print(f"Nenhum CSV encontrado em {args.output_root}/runs/*/ctx-*/ "
              "(a matriz já produziu resultados?)")
        return

    lq = aggregate_lq(rows)
    corr = correlations(rows)

    lq_cols = (["strategy", "benchmark", "num_ctx", "n", "avg_score",
                "avg_latency_ms"] + COST_COLS + QUALITY_COLS)
    write_csv(out_dir / "calibration_LQ.csv", lq, lq_cols)
    write_csv(out_dir / "calibration_corr.csv", corr,
              ["scope", "metric", "pearson_vs_score", "n"])

    # ---- Resumo legível ----
    print(f"\nLinhas agregadas: {len(rows)} testes -> {len(lq)} grupos "
          f"(estrategia x benchmark x ctx)\n")
    print("== L/Q por estrategia (media sobre benchmarks e contextos) ==")
    by_strat: Dict[str, List[Dict]] = defaultdict(list)
    for rec in lq:
        by_strat[rec["strategy"]].append(rec)
    header = f"{'strategy':<20} {'ratio(L)':>9} {'sem_fid':>8} {'evid_rec':>9} {'density':>8} {'score':>7}"
    print(header)
    print("-" * len(header))
    for strat, recs in sorted(by_strat.items()):
        print(f"{strat:<20} "
              f"{_fmt(_mean([r['cost_compression_ratio'] for r in recs]), 3):>9} "
              f"{_fmt(_mean([r['quality_semantic_fidelity'] for r in recs]), 3):>8} "
              f"{_fmt(_mean([r['quality_evidence_recall'] for r in recs]), 3):>9} "
              f"{_fmt(_mean([r['quality_info_density'] for r in recs]), 3):>8} "
              f"{_fmt(_mean([r['avg_score'] for r in recs]), 3):>7}")

    print("\n== Poder preditivo: corr(metrica, score) — apenas casos comprimidos ==")
    for rec in corr:
        if rec["scope"] == "compressed_only":
            print(f"  {rec['metric']:<28} r={_fmt(rec['pearson_vs_score'], 3):>7} "
                  f"(n={rec['n']})")

    print(f"\nCSVs: {out_dir}/calibration_LQ.csv, {out_dir}/calibration_corr.csv")


if __name__ == "__main__":
    main()
