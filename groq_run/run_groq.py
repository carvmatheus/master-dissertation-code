#!/usr/bin/env python3
"""
Groq Benchmark Runner

Executa todos os benchmarks de contexto longo com os modelos Groq disponíveis,
com retry automático em rate limit (backoff exponencial + parse do "retry-after").

Modelos disponíveis:
  - llama-3.1-8b-instant        (128K contexto)
  - llama-3.3-70b-versatile     (128K contexto)
  - openai/gpt-oss-120b         (~8K contexto empírico)
  - openai/gpt-oss-20b          (~8K contexto empírico)

Estrutura de saída:
  groq_results/{model}/{benchmark}/{strategy}.json
  groq_results/{model}/{benchmark}/{strategy}.csv
  groq_results/{model}/{benchmark}/summary.json

Uso:
  python run_groq.py
  python run_groq.py --model llama-3.3-70b-versatile
  python run_groq.py --model llama-3.3-70b-versatile --benchmark babilong
  python run_groq.py --strategies raw,sliding_window
  python run_groq.py --quick
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "01-context-extension-comparison"

sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from benchmarks import (
    NeedleInHaystackBenchmark,
    RulerBenchmark,
    LongBenchTasks,
    BABILongBenchmark,
    NarrativeQABenchmark,
    QASPERBenchmark,
    InfiniteBenchBenchmark,
)
from context_strategies import SlidingWindowStrategy, ParallelWindowStrategy, RIGStrategy
from prompt_compression import GroqSemanticCompressor
from rsaw import RSAWStrategy

# ---------------------------------------------------------------------------
# Modelos e limites de contexto
# ---------------------------------------------------------------------------
MODELS = {
    "llama-3.1-8b-instant": {
        "char_limit": 288_000,   # 128K tokens × 3 chars/token × 75%
        "short": "llama3.1-8b",
    },
    "llama-3.3-70b-versatile": {
        "char_limit": 288_000,
        "short": "llama3.3-70b",
    },
    "openai/gpt-oss-120b": {
        "char_limit": 24_000,    # ~8K tokens (limite empírico no Groq)
        "short": "gpt-oss-120b",
    },
    "openai/gpt-oss-20b": {
        "char_limit": 24_000,
        "short": "gpt-oss-20b",
    },
}

DEFAULT_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

# ---------------------------------------------------------------------------
# Cliente Groq + retry com parse de "retry-after"
# ---------------------------------------------------------------------------
def _parse_retry_after(e: Exception) -> float | None:
    """Extrai o tempo de espera sugerido pelo Groq no erro 429."""
    match = re.search(r'try again in ((?:\d+h)?(?:\d+m)?(?:[0-9.]+s)?)', str(e))
    if not match:
        return None
    total = 0.0
    for val, unit in re.findall(r'([0-9.]+)(h|m|s)', match.group(1)):
        v = float(val)
        if unit == 'h':   total += v * 3600
        elif unit == 'm': total += v * 60
        else:             total += v
    return total + 2.0 if total > 0 else None


def create_groq_llm(model_id: str, char_limit: int) -> Callable[[str, str], str]:
    """Retorna função (context, query) -> response para o modelo Groq indicado."""
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY não definida no ambiente")
    client = Groq(api_key=api_key)

    def _call(context: str, query: str) -> str:
        if len(context) > char_limit:
            logger.warning(f"[{model_id}] Contexto truncado: {len(context)} → {char_limit} chars")
            context = context[:char_limit]

        prompt = (
            "Baseado no contexto abaixo, responda a pergunta de forma direta e concisa.\n\n"
            f"CONTEXTO:\n{context}\n\nPERGUNTA: {query}\n\nRESPOSTA:"
        )

        base_wait = 5.0
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_id,
                    temperature=0,
                    max_tokens=500,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "rate_limit" in err.lower()
                if is_rate_limit:
                    wait = _parse_retry_after(e) or base_wait * (2 ** attempt)
                    logger.warning(f"[{model_id}] Rate limit (tentativa {attempt+1}/5). Aguardando {wait:.0f}s…")
                    time.sleep(wait)
                else:
                    logger.error(f"[{model_id}] Erro: {type(e).__name__}: {e}")
                    return f"[Erro: {e}]"
        logger.error(f"[{model_id}] Rate limit persistente após 5 tentativas.")
        return "[Erro: rate limit]"

    return _call


# ---------------------------------------------------------------------------
# Factories de estratégia
# ---------------------------------------------------------------------------
def build_strategies(model_id: str, char_limit: int) -> Dict[str, Callable]:
    """
    Constrói todas as estratégias para o modelo dado.
    Inclui semantic_compression (usa Groq compressor).
    """
    base = create_groq_llm(model_id, char_limit)

    config_path = SRC_DIR / "rsaw" / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    rsaw = RSAWStrategy(
        theta_alto=cfg["theta_alto"],
        theta_baixo=cfg["theta_baixo"],
        budget_tokens=cfg["budget_tokens"],
        chunk_size=cfg["chunk_size"],
        overlap=cfg["overlap"],
        tier2_ratio=cfg["tier2_ratio"],
        top_k=cfg["top_k"],
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        gamma=cfg["gamma"],
        summarizer_model=model_id,
    )
    slider    = SlidingWindowStrategy(chunk_size=500, overlap=50)
    parallel  = ParallelWindowStrategy(chunk_size=1000)
    rig       = RIGStrategy(top_k=3, alpha=0.7, beta=0.2, gamma=0.1)
    compressor = GroqSemanticCompressor(model_name=model_id)

    def sliding_fn(context: str, query: str) -> str:
        chunks = slider.process(context, query)
        max_chunks = 6
        if len(chunks) > max_chunks:
            idxs = [int(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
            chunks = [chunks[i] for i in idxs]
        return base("\n---\n".join(chunks), query)

    def parallel_fn(context: str, query: str) -> str:
        chunks = parallel.process(context, query)
        max_chunks = 4
        if len(chunks) > max_chunks:
            idxs = [int(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
            chunks = [chunks[i] for i in idxs]
        return base("\n---\n".join(chunks), query)

    def rig_fn(context: str, query: str) -> str:
        chunks = rig.process(context, query)
        return base("\n---\n".join(chunks) if chunks else context, query)

    def rsaw_fn(context: str, query: str) -> str:
        chunks = rsaw.process(context, query)
        return base(chunks[0] if chunks else context, query)

    def semantic_fn(context: str, query: str) -> str:
        compressed = compressor.compress(context, compression_ratio=0.4)
        return base(compressed, query)

    return {
        "raw":                  base,
        "sliding_window":       sliding_fn,
        "parallel_window":      parallel_fn,
        "rig":                  rig_fn,
        "rsaw":                 rsaw_fn,
        "semantic_compression": semantic_fn,
    }


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------
def save_results(results, output_dir: Path, strategy_name: str):
    """Salva resultados de uma estratégia em JSON e CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / strategy_name

    data = [r.to_dict() for r in results]
    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if data:
        excluded = {"response", "needle_fact"}
        fieldnames = ["benchmark", "strategy", "test_case", "score", "latency_ms"]
        extra_keys = sorted({k for r in data for k in r.get("details", {}) if k not in excluded})
        fieldnames += extra_keys

        with open(f"{base}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in data:
                flat = {**row}
                for k in extra_keys:
                    flat[k] = row.get("details", {}).get(k, "")
                writer.writerow(flat)


def save_summary(all_results, output_dir: Path):
    """Salva sumário agregado por estratégia."""
    from collections import defaultdict
    by_strategy = defaultdict(list)
    for r in all_results:
        by_strategy[r.strategy_name].append(r)

    summary = {}
    for strat, results in by_strategy.items():
        scores    = [r.score for r in results]
        latencies = [r.latency_ms for r in results]
        summary[strat] = {
            "avg_score":      round(sum(scores) / len(scores), 4),
            "min_score":      round(min(scores), 4),
            "max_score":      round(max(scores), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "n_tests":        len(results),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "strategies": summary}, f, indent=2)

    print(f"\n  {'Estratégia':<25} {'Score médio':>12} {'Latência (ms)':>14} {'N':>5}")
    print("  " + "-" * 60)
    for strat, m in sorted(summary.items(), key=lambda x: -x[1]["avg_score"]):
        print(f"  {strat:<25} {m['avg_score']:>12.4f} {m['avg_latency_ms']:>14.1f} {m['n_tests']:>5}")


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------
ALL_BENCHMARKS = {
    "needle_in_haystack": NeedleInHaystackBenchmark,
    "ruler":              RulerBenchmark,
    "longbench":          LongBenchTasks,
    "babilong":           BABILongBenchmark,
    "narrativeqa":        NarrativeQABenchmark,
    "qasper":             QASPERBenchmark,
    "infinitebench":      InfiniteBenchBenchmark,
}

QUICK_CONFIGS = {
    "needle_in_haystack": {"num_paragraphs": 10, "num_needles": 2, "positions": ["start", "middle", "end"]},
    "ruler":              {"context_sizes": [10, 25], "num_facts_per_context": 2},
    "longbench":          {"num_qa_cases": 3},
    "babilong":           {"context_lengths": ["4k", "8k"], "tasks": ["qa1"], "num_examples_per_config": 2},
    "narrativeqa":        {"num_examples": 5},
    "qasper":             {"num_examples": 5},
    "infinitebench":      {"task": "En.QA", "num_examples": 5},
}

FULL_CONFIGS = {
    "needle_in_haystack": {"num_paragraphs": 20, "num_needles": 3, "positions": ["start", 0.25, "middle", 0.75, "end"]},
    "ruler":              {"context_sizes": [10, 25, 50], "num_facts_per_context": 3},
    "longbench":          {"num_qa_cases": 5},
    "babilong":           {"context_lengths": ["4k", "8k", "16k", "32k"], "tasks": ["qa1", "qa2"], "num_examples_per_config": 3},
    "narrativeqa":        {"num_examples": 20},
    "qasper":             {"num_examples": 20},
    "infinitebench":      {"task": "En.QA", "num_examples": 10},
}


def run_model_benchmark(
    model_id: str,
    model_short: str,
    benchmark_name: str,
    strategies: Dict[str, Callable],
    bench_kwargs: dict,
    results_root: Path,
):
    """Roda um benchmark com todas as estratégias para um modelo e salva resultados."""
    output_dir = results_root / model_short / benchmark_name

    print(f"\n{'='*70}")
    print(f"  Modelo    : {model_id}")
    print(f"  Benchmark : {benchmark_name}")
    print(f"  Estratégias: {', '.join(strategies.keys())}")
    print(f"  Saída     : {output_dir}")
    print(f"{'='*70}")

    benchmark = ALL_BENCHMARKS[benchmark_name]()
    all_results = []

    for strat_name, strat_fn in strategies.items():
        print(f"\n  >> Estratégia: {strat_name}")
        try:
            results = benchmark.run_all(
                strategy_fn=strat_fn,
                strategy_name=strat_name,
                **bench_kwargs,
            )
            all_results.extend(results)

            avg_score = sum(r.score for r in results) / len(results) if results else 0
            avg_lat   = sum(r.latency_ms for r in results) / len(results) if results else 0
            print(f"     Score médio: {avg_score:.4f} | Latência média: {avg_lat:.1f}ms | {len(results)} casos")

            save_results(results, output_dir, strat_name)
        except Exception as e:
            print(f"     ERRO em '{strat_name}': {type(e).__name__}: {e}")
            logger.exception(e)

    if all_results:
        save_summary(all_results, output_dir)
        print(f"\n  Resultados salvos em: {output_dir}")
    else:
        print("  Nenhum resultado gerado.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa benchmarks de contexto longo com modelos Groq"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Modelo a testar (padrão: all)",
    )
    parser.add_argument(
        "--benchmark",
        choices=list(ALL_BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Benchmark a executar (padrão: all)",
    )
    parser.add_argument(
        "--strategies",
        default="all",
        help="Estratégias: raw,sliding_window,parallel_window,rig,rsaw,semantic_compression ou all",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "groq_results"),
        help="Diretório raiz para resultados",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido com menos casos de teste",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.output_dir)

    models_to_run     = DEFAULT_MODELS if args.model == "all" else [args.model]
    benchmarks_to_run = list(ALL_BENCHMARKS.keys()) if args.benchmark == "all" else [args.benchmark]
    bench_configs     = QUICK_CONFIGS if args.quick else FULL_CONFIGS

    all_strategies = ["raw", "sliding_window", "parallel_window", "rig", "rsaw", "semantic_compression"]
    strategy_names = all_strategies if args.strategies == "all" else [s.strip() for s in args.strategies.split(",")]

    print("=" * 70)
    print("GROQ — BENCHMARK DE EXTENSÃO DE CONTEXTO")
    print("=" * 70)
    print(f"Modelos    : {', '.join(models_to_run)}")
    print(f"Benchmarks : {', '.join(benchmarks_to_run)}")
    print(f"Estratégias: {', '.join(strategy_names)}")
    print(f"Modo       : {'rápido' if args.quick else 'completo'}")

    for model_id in models_to_run:
        cfg         = MODELS[model_id]
        model_short = cfg["short"]
        char_limit  = cfg["char_limit"]

        print(f"\n\n{'#'*70}")
        print(f"# Carregando estratégias para: {model_id}")
        print(f"{'#'*70}")

        try:
            all_strats = build_strategies(model_id, char_limit)
            strategies = {k: v for k, v in all_strats.items() if k in strategy_names}
        except Exception as e:
            print(f"ERRO ao inicializar modelo '{model_id}': {e}")
            continue

        for benchmark_name in benchmarks_to_run:
            bench_kwargs = bench_configs.get(benchmark_name, {})
            run_model_benchmark(
                model_id=model_id,
                model_short=model_short,
                benchmark_name=benchmark_name,
                strategies=strategies,
                bench_kwargs=bench_kwargs,
                results_root=results_root,
            )

    print("\n\n" + "=" * 70)
    print("CONCLUÍDO")
    print(f"Resultados em: {results_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
