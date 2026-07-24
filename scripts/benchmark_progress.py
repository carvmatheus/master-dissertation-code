#!/usr/bin/env python3
"""
Barra de progresso da matriz de benchmarks.

Lê o log do run (linhas ">> Executando <benchmark> com estratégia: <strategy>")
e mostra quanto já rodou e quanto falta, no total e por contexto.

Uso:
  python scripts/benchmark_progress.py --log <arquivo.log> \
      --models gemma4:26b-mlx,qwen3:30b-a3b \
      --contexts 8192,16384,32768 \
      --benchmarks longbench,zeroscrolls,naturalquestions,triviaqa,hotpotqa,musique,meeting_summarization \
      --strategies raw,sliding_window,parallel_window,semantic_compression,rig

Sem dependências além da stdlib.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

EXEC_RE = re.compile(r">> Executando (\S+) com estratégia: (\S+)")
# O contexto atual vem da linha do processo filho "Carregando ... (num_ctx=N)"
# ou, quando não bufferizada, da linha "RUN ... num_ctx=N" do processo pai.
RUN_RE = re.compile(r"num_ctx=(\d+)")
MODEL_RUN_RE = re.compile(r"RUN (\S+) com num_ctx=(\d+)")
MODEL_LOAD_RE = re.compile(r"Carregando (\S+) \(num_ctx=(\d+)\)")
DONE_RE = re.compile(r"Comparação salva em:")


def _bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    frac = min(done / total, 1.0)
    filled = int(round(frac * width))
    return "[" + "█" * filled + "·" * (width - filled) + f"] {frac*100:5.1f}%"


def render(
    log: Path,
    contexts: list[str],
    n_bench: int,
    n_strat: int,
    stamp: str = "",
    models: list[str] | None = None,
) -> tuple[str, bool]:
    """Monta o texto da barra a partir do log. Retorna (texto, concluído)."""
    per_ctx = n_bench * n_strat
    total = per_ctx * len(contexts) * (len(models) if models else 1)

    if not log.exists():
        return f"Log ainda não existe: {log}", False

    text = log.read_text(errors="replace")

    current_model = None
    current_ctx = None
    per_ctx_done: Counter = Counter()
    seen_ctx_order: list[str] = []
    per_model_ctx_done: Counter = Counter()
    seen_model_contexts: set[tuple[str, str]] = set()
    for line in text.splitlines():
        model_match = MODEL_LOAD_RE.search(line) or MODEL_RUN_RE.search(line)
        if model_match:
            current_model, current_ctx = model_match.groups()
            seen_model_contexts.add((current_model, current_ctx))
            if current_ctx not in seen_ctx_order:
                seen_ctx_order.append(current_ctx)
            continue
        if EXEC_RE.search(line):
            if current_ctx is not None:
                per_ctx_done[current_ctx] += 1
            if current_model is not None and current_ctx is not None:
                per_model_ctx_done[(current_model, current_ctx)] += 1
            continue
        m = RUN_RE.search(line)
        if m:
            current_ctx = m.group(1)
            if current_ctx not in seen_ctx_order:
                seen_ctx_order.append(current_ctx)

    if models:
        selected = {(model, ctx) for model in models for ctx in contexts}
        done_total = sum(per_model_ctx_done[key] for key in selected)
    else:
        done_total = sum(per_ctx_done.values())
    finished = bool(DONE_RE.search(text)) and done_total >= total

    lines = [
        "Matriz de benchmarks — progresso" + (f"   ({stamp})" if stamp else ""),
        f"{n_bench} benchmarks × {n_strat} estratégias × {len(contexts)} contextos "
        f"= {total} execuções",
        "",
        f"TOTAL  {_bar(done_total, total)}  {done_total}/{total}",
        "",
    ]
    if models:
        for model in models:
            lines.append(model)
            for ctx in contexts:
                key = (model, ctx)
                d = per_model_ctx_done.get(key, 0)
                if key not in seen_model_contexts and d == 0:
                    status = "  (na fila)"
                elif d >= per_ctx:
                    status = "  ✓ completo"
                elif d > 0:
                    status = "  ⟳ rodando"
                else:
                    status = ""
                lines.append(
                    f"  ctx-{ctx:<6} {_bar(d, per_ctx)}  {d}/{per_ctx}{status}"
                )
    else:
        for ctx in contexts:
            d = per_ctx_done.get(ctx, 0)
            if ctx not in seen_ctx_order and d == 0:
                status = "  (na fila)"
            elif d >= per_ctx:
                status = "  ✓ completo"
            elif d > 0:
                status = "  ⟳ rodando"
            else:
                status = ""
            lines.append(f"ctx-{ctx:<6} {_bar(d, per_ctx)}  {d}/{per_ctx}{status}")

    lines.append("")
    if finished:
        lines.append("Status: CONCLUÍDO ✓")
    else:
        lines.append(f"Status: em andamento — faltam {total - done_total} execuções")
    return "\n".join(lines), finished


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument(
        "--models",
        default="",
        help="Modelos do lote separados por vírgula; ignora runs anteriores no mesmo log.",
    )
    ap.add_argument("--contexts", default="8192,16384,32768")
    ap.add_argument(
        "--benchmarks",
        default="longbench,zeroscrolls,naturalquestions,triviaqa,hotpotqa,musique,meeting_summarization",
    )
    ap.add_argument(
        "--strategies",
        default="raw,sliding_window,parallel_window,semantic_compression,rig",
    )
    ap.add_argument("--watch", type=float, default=0, metavar="SEG",
                    help="Redesenha a barra no mesmo lugar a cada SEG segundos "
                         "até concluir (Ctrl-C para sair).")
    args = ap.parse_args()

    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    models = [m.strip().removeprefix("ollama/") for m in args.models.split(",") if m.strip()]
    n_bench = len([b for b in args.benchmarks.split(",") if b.strip()])
    n_strat = len([s for s in args.strategies.split(",") if s.strip()])

    if args.watch <= 0:
        text, _ = render(args.log, contexts, n_bench, n_strat, models=models)
        print("\n" + text)
        return

    import time
    try:
        while True:
            stamp = time.strftime("%H:%M:%S")
            text, finished = render(
                args.log, contexts, n_bench, n_strat, stamp, models=models
            )
            # Limpa a tela e volta ao topo (ANSI), depois redesenha.
            print("\033[2J\033[H" + text, flush=True)
            if finished:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n(monitor encerrado)")


if __name__ == "__main__":
    main()
