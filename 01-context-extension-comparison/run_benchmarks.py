#!/usr/bin/env python3
"""
Benchmark Runner Principal

Executa todos os benchmarks de contexto longo contra as estratégias implementadas:
- Needle-in-a-Haystack
- RULER
- LongBench

Suporta múltiplos modelos Groq para comparação cruzada.

Gera arquivos de comparação em CSV e JSON.

Uso:
    python run_benchmarks.py
    python run_benchmarks.py --strategies raw,sliding_window
    python run_benchmarks.py --models llama-3.1-8b-instant,llama-3.3-70b-versatile
    python run_benchmarks.py --benchmarks needle_in_haystack,ruler
    python run_benchmarks.py --output-dir ./results
    python run_benchmarks.py --quick  # Modo rápido com menos testes
"""
import argparse
import os
import sys
from typing import List, Callable, Dict, Any
from pathlib import Path

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


# Modelos Groq disponíveis para benchmark
AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",  # modelo legado
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# Modelos padrão para testar
DEFAULT_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

# Modelos GPT-OSS
GPT_OSS_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


def create_groq_strategy(model: str) -> Callable[[str, str], str]:
    """
    Cria estratégia que usa Groq para responder perguntas.
    
    Args:
        model: Nome do modelo Groq a usar
        
    Returns:
        Função (context, query) -> response
    """
    from groq import Groq
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY não definida no ambiente")
    
    client = Groq(api_key=api_key)
    
    def strategy_fn(context: str, query: str) -> str:
        """Envia contexto + query para o Groq e retorna resposta."""
        prompt = f"""Baseado no contexto abaixo, responda a pergunta de forma direta e concisa.

CONTEXTO:
{context}

PERGUNTA: {query}

RESPOSTA:"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Erro: {e}]"
    
    return strategy_fn


def create_raw_strategy(model: str) -> Callable[[str, str], str]:
    """
    Estratégia baseline: envia contexto bruto para o LLM.
    """
    return create_groq_strategy(model)


def create_sliding_window_strategy(
    model: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> Callable[[str, str], str]:
    """
    Estratégia com janela deslizante.
    Usa apenas os primeiros chunks que cabem no contexto.
    """
    slider = SlidingWindowStrategy(chunk_size=chunk_size, overlap=overlap)
    base_fn = create_groq_strategy(model)
    
    def strategy_fn(context: str, query: str) -> str:
        chunks = slider.process(context, query)
        # Usa os 3 primeiros chunks
        limited_context = "\n---\n".join(chunks[:3])
        return base_fn(limited_context, query)
    
    return strategy_fn


def create_parallel_window_strategy(
    model: str,
    chunk_size: int = 1000
) -> Callable[[str, str], str]:
    """
    Estratégia com janela paralela.
    Processa chunks em paralelo e agrega respostas.
    """
    slider = ParallelWindowStrategy(chunk_size=chunk_size)
    base_fn = create_groq_strategy(model)
    
    def strategy_fn(context: str, query: str) -> str:
        chunks = slider.process(context, query)
        # Concatena primeiros chunks
        limited_context = "\n---\n".join(chunks[:3])
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
    base_fn = create_groq_strategy(model)
    
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
    base_fn = create_groq_strategy(model)
    
    def strategy_fn(context: str, query: str) -> str:
        chunks = rig.process(context, query)
        if not chunks:
            return base_fn(context, query)
        
        combined_context = "\n---\n".join(chunks)
        return base_fn(combined_context, query)
    
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
        "llama-3.1-8b-instant": "llama3.1-8b",
        "llama-3.3-70b-versatile": "llama3.3-70b",
        "llama3-70b-8192": "llama3-70b",
        "openai/gpt-oss-120b": "gpt-oss-120b",
        "openai/gpt-oss-20b": "gpt-oss-20b",
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
        help=f"Modelos Groq a testar (comma-separated). Disponíveis: {', '.join(AVAILABLE_MODELS)}"
    )
    
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Estratégias a testar (comma-separated). Opções: raw, sliding_window, parallel_window, semantic_compression, rig, mock, all"
    )
    
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Benchmarks a executar (comma-separated). Opções: needle_in_haystack, ruler, longbench, all"
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
            "semantic_compression", "rig"
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
                    print(f"  AVISO: Não foi possível registrar '{strategy_name}': {e}")
    
    if not runner.strategies:
        print("ERRO: Nenhuma estratégia registrada. Verifique GROQ_API_KEY ou use --mock-only")
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
