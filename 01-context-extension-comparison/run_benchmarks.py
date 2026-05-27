#!/usr/bin/env python3
"""
Benchmark Runner Principal

Executa todos os benchmarks de contexto longo contra as estratégias implementadas:
- Needle-in-a-Haystack
- RULER
- LongBench
- BABILong, NarrativeQA, QASPER, InfiniteBench

Suporta múltiplos provedores de LLM para comparação cruzada:
  - Groq        (GROQ_API_KEY)       — 128K contexto, rápido
  - Gemini      (GEMINI_API_KEY)     — 1M contexto, free tier Google AI Studio
  - Cerebras    (CEREBRAS_API_KEY)   — 8K contexto free tier, altíssima velocidade

Prefixos de modelo:
  - "gemini-*"      → Google AI Studio
  - "cerebras/*"    → Cerebras Inference
  - qualquer outro  → Groq

Uso:
    python run_benchmarks.py
    python run_benchmarks.py --models gemini-2.5-flash,llama-3.3-70b-versatile
    python run_benchmarks.py --strategies raw,sliding_window
    python run_benchmarks.py --benchmarks needle_in_haystack,ruler
    python run_benchmarks.py --output-dir ./results
    python run_benchmarks.py --quick  # Modo rápido com menos testes
"""
import argparse
import logging
import os
import sys
from typing import List, Callable, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Adiciona o diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Carrega .env
load_dotenv(Path(__file__).parent.parent / ".env")

from benchmarks import BenchmarkRunner
from context_strategies import (
    SlidingWindowStrategy,
    ParallelWindowStrategy,
    RIGStrategy,
)
from prompt_compression import GroqSemanticCompressor, PerplexityCompressor
from rsaw import RSAWStrategy


# ---------------------------------------------------------------------------
# Modelos disponíveis por provedor
# ---------------------------------------------------------------------------

# Groq — contexto até 128K tokens
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# Google AI Studio (Gemini) — contexto até 1M tokens, gratuito
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Cerebras — free tier limita a 8K tokens de contexto; alta velocidade
CEREBRAS_MODELS = [
    "cerebras/llama-3.3-70b",
    "cerebras/qwen3-32b",
]

AVAILABLE_MODELS = GROQ_MODELS + GEMINI_MODELS + CEREBRAS_MODELS

# Modelos padrão para testar
DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",
    "gemini-2.5-flash",
]


# Limite de chars por modelo (≈ 75% da janela de contexto disponível)
MODEL_CONTEXT_CHAR_LIMITS = {
    # Groq
    "llama-3.1-8b-instant":    384_000,   # 128K tokens
    "llama-3.3-70b-versatile": 384_000,   # 128K tokens
    "openai/gpt-oss-120b":      24_000,   # ~8K (limite empírico no Groq)
    "openai/gpt-oss-20b":       24_000,
    # Gemini (Google AI Studio) — 1M tokens × 3 chars/token × 75%
    "gemini-2.5-flash":       2_250_000,
    "gemini-2.0-flash":       2_250_000,
    # Cerebras — free tier: 8K tokens
    "cerebras/llama-3.3-70b":   18_000,
    "cerebras/qwen3-32b":       18_000,
}
DEFAULT_CONTEXT_CHAR_LIMIT = 96_000  # fallback seguro


def create_openai_compatible_strategy(
    model: str,
    base_url: str,
    api_key_env: str,
    char_limit: int,
    model_id: str | None = None,
) -> Callable[[str, str], str]:
    """
    Estratégia genérica para APIs compatíveis com OpenAI (Gemini, Cerebras, etc.).

    Args:
        model:       nome lógico do modelo (usado em logs)
        base_url:    endpoint OpenAI-compatible do provedor
        api_key_env: nome da variável de ambiente com a API key
        char_limit:  máximo de caracteres no contexto
        model_id:    ID real do modelo para a API (padrão: igual a model)
    """
    import time
    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} não definida no ambiente")

    client = OpenAI(api_key=api_key, base_url=base_url)
    actual_model = model_id or model

    def strategy_fn(context: str, query: str) -> str:
        if len(context) > char_limit:
            logger.warning(f"[{model}] Contexto truncado: {len(context)} → {char_limit} chars")
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
                    model=actual_model,
                    temperature=0,
                    max_tokens=500,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower()
                if is_rate_limit:
                    wait = base_wait * (2 ** attempt)
                    logger.warning(f"[{model}] Rate limit (tentativa {attempt+1}/5). Aguardando {wait:.0f}s…")
                    time.sleep(wait)
                else:
                    logger.error(f"[{model}] Erro na API: {type(e).__name__}: {e}")
                    return f"[Erro: {e}]"
        logger.error(f"[{model}] Rate limit persistente após 5 tentativas.")
        return "[Erro: rate limit]"

    return strategy_fn


def create_gemini_strategy(model: str) -> Callable[[str, str], str]:
    """Estratégia para Google AI Studio (Gemini) — contexto até 1M tokens."""
    return create_openai_compatible_strategy(
        model=model,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        char_limit=MODEL_CONTEXT_CHAR_LIMITS.get(model, DEFAULT_CONTEXT_CHAR_LIMIT),
    )


def create_cerebras_strategy(model: str) -> Callable[[str, str], str]:
    """Estratégia para Cerebras Inference — free tier limitado a 8K tokens."""
    # Remove prefixo "cerebras/" para o ID real na API
    model_id = model.removeprefix("cerebras/")
    return create_openai_compatible_strategy(
        model=model,
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        char_limit=MODEL_CONTEXT_CHAR_LIMITS.get(model, DEFAULT_CONTEXT_CHAR_LIMIT),
        model_id=model_id,
    )


def _route_api_strategy(model: str) -> Callable[[str, str], str]:
    """Seleciona a função de estratégia correta de acordo com o prefixo do modelo."""
    if model.startswith("gemini-"):
        return create_gemini_strategy(model)
    if model.startswith("cerebras/"):
        return create_cerebras_strategy(model)
    return create_groq_strategy(model)


def create_groq_strategy(model: str) -> Callable[[str, str], str]:
    """Cria estratégia base Groq com retry, truncação e log de erros."""
    import time
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY não definida no ambiente")

    client = Groq(api_key=api_key)
    char_limit = MODEL_CONTEXT_CHAR_LIMITS.get(model, DEFAULT_CONTEXT_CHAR_LIMIT)

    def _parse_retry_after(e: Exception) -> float:
        """Extrai o tempo de espera sugerido pelo Groq no erro 429.
        Formatos: '12.3s', '1m30s', '1h2m3.4s'
        """
        import re
        err_str = str(e)
        match = re.search(r'try again in ((?:\d+h)?(?:\d+m)?(?:[0-9.]+s)?)', err_str)
        if not match:
            return None
        raw = match.group(1)
        total = 0.0
        for val, unit in re.findall(r'([0-9.]+)(h|m|s)', raw):
            v = float(val)
            if unit == 'h':
                total += v * 3600
            elif unit == 'm':
                total += v * 60
            else:
                total += v
        return total + 2.0 if total > 0 else None

    def strategy_fn(context: str, query: str) -> str:
        # Trunca contexto se exceder limite do modelo
        if len(context) > char_limit:
            logger.warning(
                f"[{model}] Contexto truncado: {len(context)} → {char_limit} chars"
            )
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
                    model=model,
                    temperature=0,
                    max_tokens=500,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                if is_rate_limit:
                    wait = _parse_retry_after(e)
                    if wait is None:
                        wait = base_wait * (2 ** attempt)  # 5, 10, 20, 40, 80s
                    logger.warning(
                        f"[{model}] Rate limit (tentativa {attempt+1}/5). Aguardando {wait:.0f}s… | msg: {err_str[:200]}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"[{model}] Erro na API: {type(e).__name__}: {e}")
                    return f"[Erro: {e}]"
        logger.error(f"[{model}] Rate limit persistente após 5 tentativas.")
        return "[Erro: rate limit]"

    return strategy_fn


def create_raw_strategy(model: str) -> Callable[[str, str], str]:
    """Estratégia baseline: envia contexto bruto para o LLM."""
    return _route_api_strategy(model)


def create_sliding_window_strategy(
    model: str,
    chunk_size: int = 500,
    overlap: int = 50,
    max_chunks: int = 6,
) -> Callable[[str, str], str]:
    """
    Estratégia com janela deslizante.
    Distribui chunks ao longo do documento (início, meio, fim) para cobrir
    contextos longos sem se limitar apenas ao começo.
    """
    slider = SlidingWindowStrategy(chunk_size=chunk_size, overlap=overlap)
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> str:
        chunks = slider.process(context, query)
        if len(chunks) <= max_chunks:
            selected = chunks
        else:
            # Distribui uniformemente: início, meio e fim
            indices = [int(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
            selected = [chunks[i] for i in indices]
        limited_context = "\n---\n".join(selected)
        return base_fn(limited_context, query)

    return strategy_fn


def create_parallel_window_strategy(
    model: str,
    chunk_size: int = 1000,
    max_chunks: int = 4,
) -> Callable[[str, str], str]:
    """
    Estratégia com janela paralela.
    Distribui chunks ao longo do documento para cobrir contextos longos.
    """
    slider = ParallelWindowStrategy(chunk_size=chunk_size)
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> str:
        chunks = slider.process(context, query)
        if len(chunks) <= max_chunks:
            selected = chunks
        else:
            indices = [int(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
            selected = [chunks[i] for i in indices]
        limited_context = "\n---\n".join(selected)
        return base_fn(limited_context, query)

    return strategy_fn


def create_semantic_compression_strategy(
    model: str,
    compression_ratio: float = 0.4
) -> Callable[[str, str], str]:
    """
    Estratégia com compressão semântica via LLM.
    """
    compressor = GroqSemanticCompressor(model_name=model)
    base_fn = _route_api_strategy(model)
    
    def strategy_fn(context: str, query: str) -> str:
        compressed = compressor.compress(context, compression_ratio)
        return base_fn(compressed, query)
    
    return strategy_fn


def create_rig_strategy(
    model: str,
    top_k: int = 3,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1
) -> Callable[[str, str], str]:
    """
    Estratégia RIG com Dartboard ranking.
    """
    rig = RIGStrategy(top_k=top_k, alpha=alpha, beta=beta, gamma=gamma)
    base_fn = _route_api_strategy(model)
    
    def strategy_fn(context: str, query: str) -> str:
        chunks = rig.process(context, query)
        if not chunks:
            return base_fn(context, query)
        
        combined_context = "\n---\n".join(chunks)
        return base_fn(combined_context, query)
    
    return strategy_fn


def create_rsaw_strategy(model: str) -> Callable[[str, str], str]:
    """
    Estratégia RSAW: Relevance-Stratified Adaptive Window.
    Lê hiperparâmetros de rsaw/config.json.
    """
    import json

    config_path = Path(__file__).parent / "rsaw" / "config.json"
    with open(config_path, "r") as f:
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
        summarizer_model=model,
    )
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> str:
        chunks = rsaw.process(context, query)
        if not chunks:
            return base_fn(context, query)
        return base_fn(chunks[0], query)

    return strategy_fn


def create_mock_strategy() -> Callable[[str, str], str]:
    """
    Estratégia mock para testes (não usa API).
    Retorna resposta baseada em busca simples no contexto.
    """
    def strategy_fn(context: str, query: str) -> str:
        # Busca simples: procura palavras da query no contexto
        query_words = query.lower().split()
        
        # Divide contexto em sentenças
        sentences = context.replace("\n", " ").split(".")
        
        # Pontua sentenças por overlap com query
        scored = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for w in query_words if w in sent_lower)
            scored.append((score, sent.strip()))
        
        # Retorna a sentença mais relevante
        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]
        
        return "Informação não encontrada no contexto."
    
    return strategy_fn


def get_model_short_name(model: str) -> str:
    """Retorna nome curto do modelo para usar em identificadores."""
    # Remove prefixos comuns
    short = model.replace("openai/", "").replace("meta-llama/", "")
    
    # Simplifica nomes longos
    replacements = {
        "llama-3.1-8b-instant":    "llama3.1-8b",
        "llama-3.3-70b-versatile": "llama3.3-70b",
        "llama3-70b-8192":         "llama3-70b",
        "openai/gpt-oss-120b":     "gpt-oss-120b",
        "openai/gpt-oss-20b":      "gpt-oss-20b",
        "gemini-2.5-flash":        "gemini-2.5-flash",
        "gemini-2.0-flash":        "gemini-2.0-flash",
        "cerebras/llama-3.3-70b":  "cb-llama3.3-70b",
        "cerebras/qwen3-32b":      "cb-qwen3-32b",
    }
    
    return replacements.get(model, short)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa benchmarks de contexto longo com múltiplos modelos"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=(
            f"Modelos a testar (comma-separated). Disponíveis: {', '.join(AVAILABLE_MODELS)}. "
            "Prefixo 'gemini-*' usa GEMINI_API_KEY; 'cerebras/*' usa CEREBRAS_API_KEY; demais usam GROQ_API_KEY."
        )
    )
    
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Estratégias a testar (comma-separated). Opções: raw, sliding_window, parallel_window, semantic_compression, rig, rsaw, mock, all"
    )
    
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Benchmarks a executar (comma-separated). Opções: needle_in_haystack, ruler, longbench, babilong, narrativeqa, qasper, infinitebench, all"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Diretório para salvar resultados"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido com menos casos de teste"
    )
    
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Usa apenas estratégia mock (não requer API key)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("BENCHMARK DE ESTRATÉGIAS DE EXTENSÃO DE CONTEXTO")
    print("=" * 70)
    
    # Cria runner
    runner = BenchmarkRunner(output_dir=args.output_dir)
    
    # Parse modelos
    if args.mock_only:
        models_to_test = []
    else:
        models_to_test = [m.strip() for m in args.models.split(",")]
    
    # Parse estratégias
    if args.mock_only:
        strategy_types = ["mock"]
    elif args.strategies == "all":
        strategy_types = [
            "raw", "sliding_window", "parallel_window",
            "semantic_compression", "rig", "rsaw"
        ]
    else:
        strategy_types = [s.strip() for s in args.strategies.split(",")]
    
    # Factory functions para cada tipo de estratégia
    strategy_factories = {
        "raw": lambda m: create_raw_strategy(m),
        "sliding_window": lambda m: create_sliding_window_strategy(m),
        "parallel_window": lambda m: create_parallel_window_strategy(m),
        "semantic_compression": lambda m: create_semantic_compression_strategy(m),
        "rig": lambda m: create_rig_strategy(m),
        "rsaw": lambda m: create_rsaw_strategy(m),
    }
    
    # Registra estratégias para cada modelo
    print("\nRegistrando estratégias...")
    
    if args.mock_only:
        runner.register_strategy("mock", create_mock_strategy(), "Mock (sem API)")
    else:
        for model in models_to_test:
            model_short = get_model_short_name(model)
            
            for strat_type in strategy_types:
                if strat_type == "mock":
                    continue
                    
                if strat_type not in strategy_factories:
                    print(f"  AVISO: Estratégia desconhecida: {strat_type}")
                    continue
                
                # Nome composto: estrategia_modelo
                strategy_name = f"{strat_type}_{model_short}"
                
                try:
                    strategy_fn = strategy_factories[strat_type](model)
                    description = f"{strat_type} com {model}"
                    runner.register_strategy(strategy_name, strategy_fn, description)
                except Exception as e:
                    import traceback
                    print(f"  AVISO: Não foi possível registrar '{strategy_name}': {type(e).__name__}: {e}")
                    traceback.print_exc()
    
    if not runner.strategies:
        print("ERRO: Nenhuma estratégia registrada. Verifique GROQ_API_KEY / GEMINI_API_KEY / CEREBRAS_API_KEY ou use --mock-only")
        sys.exit(1)
    
    # Configuração dos benchmarks
    if args.quick:
        benchmark_configs = {
            "needle_in_haystack": {
                "num_paragraphs": 10,
                "num_needles": 2,
                "positions": ["start", "middle", "end"],
            },
            "ruler": {
                "context_sizes": [10, 25],
                "num_facts_per_context": 2,
            },
            "longbench": {
                "num_qa_cases": 3,
            },
            "babilong": {
                "context_lengths": ["4k", "8k"],
                "tasks": ["qa1"],
                "num_examples_per_config": 2,
            },
            "narrativeqa": {
                "num_examples": 5,
            },
            "qasper": {
                "num_examples": 5,
            },
            "infinitebench": {
                "task": "En.QA",
                "num_examples": 5,
            },
        }
    else:
        benchmark_configs = {
            "needle_in_haystack": {
                "num_paragraphs": 20,
                "num_needles": 3,
                "positions": ["start", 0.25, "middle", 0.75, "end"],
            },
            "ruler": {
                "context_sizes": [10, 25, 50],
                "num_facts_per_context": 3,
            },
            "longbench": {
                "num_qa_cases": 5,
            },
            "babilong": {
                "context_lengths": ["4k", "8k", "16k", "32k"],
                "tasks": ["qa1", "qa2"],
                "num_examples_per_config": 3,
            },
            "narrativeqa": {
                "num_examples": 10,
            },
            "qasper": {
                "num_examples": 10,
            },
            "infinitebench": {
                "task": "En.QA",
                "num_examples": 10,
            },
        }
    
    # Filtra benchmarks se especificado
    if args.benchmarks != "all":
        selected = [b.strip() for b in args.benchmarks.split(",")]
        benchmark_configs = {
            k: v for k, v in benchmark_configs.items()
            if k in selected
        }
    
    # Executa benchmarks
    print(f"\nModelos: {models_to_test if models_to_test else ['mock']}")
    print(f"Estratégias base: {strategy_types}")
    print(f"Estratégias registradas: {list(runner.strategies.keys())}")
    print(f"Benchmarks: {list(benchmark_configs.keys())}")
    print("-" * 70)
    
    runner.run_all_benchmarks(benchmark_configs)
    
    # Salva resultados
    print("\nSalvando resultados...")
    runner.save_results_csv()
    runner.save_results_json()
    runner.save_comparison_table()
    
    # Gera tabela adicional: modelo x benchmark
    save_model_comparison(runner, args.output_dir)
    
    # Imprime sumário
    runner.print_summary()
    
    print(f"\nResultados salvos em: {args.output_dir}/")
    print("  - benchmark_results.csv         (todos os resultados)")
    print("  - benchmark_results.json        (relatório completo)")
    print("  - comparison_table.csv          (estratégia × benchmark)")
    print("  - model_comparison.csv          (modelo × benchmark)")


def save_model_comparison(runner: BenchmarkRunner, output_dir: str) -> None:
    """
    Salva tabela de comparação modelo × benchmark.
    """
    from collections import defaultdict
    import csv
    
    output_path = Path(output_dir) / "model_comparison.csv"
    
    if not runner.results:
        return
    
    # Extrai modelo do nome da estratégia
    # Ex: "raw_llama3.1-8b" -> "llama3.1-8b"
    model_scores = defaultdict(lambda: defaultdict(list))
    
    for r in runner.results:
        parts = r.strategy_name.rsplit("_", 1)
        if len(parts) == 2:
            model = parts[1]
        else:
            model = r.strategy_name
        
        model_scores[model][r.benchmark_name].append(r.score)
    
    # Calcula médias
    benchmarks = list(runner.benchmarks.keys())
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Model"] + benchmarks + ["Average"])
        
        for model, bench_scores in model_scores.items():
            row = [model]
            all_scores = []
            
            for bench in benchmarks:
                scores = bench_scores.get(bench, [])
                if scores:
                    avg = sum(scores) / len(scores)
                    row.append(f"{avg:.3f}")
                    all_scores.append(avg)
                else:
                    row.append("-")
            
            # Average
            if all_scores:
                row.append(f"{sum(all_scores) / len(all_scores):.3f}")
            else:
                row.append("-")
            
            writer.writerow(row)
    
    print(f"  - model_comparison.csv          (modelo × benchmark)")


if __name__ == "__main__":
    main()
