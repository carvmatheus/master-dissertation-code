#!/usr/bin/env python3
"""
Benchmark Runner Principal

Executa todos os benchmarks de contexto longo contra as estratégias implementadas:
- Needle-in-a-Haystack
- RULER
- LongBench
- BABILong, NarrativeQA, QASPER, InfiniteBench

Roda contra modelos locais via Ollama (http://localhost:11434).
Modelos usam o prefixo "ollama/", removido antes da chamada à API.

Uso:
    python run_benchmarks.py
    python run_benchmarks.py --models ollama/llama3.1:8b-instruct-q8_0
    python run_benchmarks.py --strategies raw,sliding_window
    python run_benchmarks.py --benchmarks needle_in_haystack,ruler
    python run_benchmarks.py --output-dir ./results
    python run_benchmarks.py --quick  # Modo rápido com menos testes
"""
import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
from benchmarks.base import StrategyOutput
from context_strategies import (
    SlidingWindowStrategy,
    ParallelWindowStrategy,
    RIGStrategy,
)
from prompt_compression import OllamaSemanticCompressor, PerplexityCompressor
from rsaw import RSAWStrategy
from mckp import MCKPStrategy, MCKPConfig


# ---------------------------------------------------------------------------
# Modelos disponíveis (Ollama local)
# ---------------------------------------------------------------------------

OLLAMA_MODELS = [
    "ollama/hf.co/mradermacher/Llama-4-Scout-17B-6E-Instruct-GGUF:Q4_K_S",
    "ollama/gpt-oss:20b",
    "ollama/llama3.1:8b-instruct-q8_0",
    "ollama/gemma4:26b-mlx",
    "ollama/qwen3.6:35b-mlx",
    "ollama/llama3.1:8b-text-q4_K_M",
    "ollama/deepseek-r1:32b",
]

AVAILABLE_MODELS = OLLAMA_MODELS

# Modelos padrão para testar
DEFAULT_MODELS = [
    "ollama/llama3.1:8b-instruct-q8_0",
]


def _ollama_api_base() -> str:
    """Retorna a raiz da API nativa mesmo se OLLAMA_BASE_URL terminar em /v1."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return base_url.removesuffix("/v1").rstrip("/")


def _ollama_request(path: str, payload: Dict[str, Any], timeout: int = 900) -> Dict[str, Any]:
    request = urllib.request.Request(
        f"{_ollama_api_base()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def set_ollama_model_loaded(model: str, num_ctx: int, loaded: bool) -> None:
    """Carrega um modelo com a janela pedida ou o remove imediatamente da memória."""
    ollama_model = model.removeprefix("ollama/")
    payload: Dict[str, Any] = {
        "model": ollama_model,
        "prompt": "",
        "stream": False,
        "keep_alive": -1 if loaded else 0,
    }
    if loaded:
        payload["options"] = {"num_ctx": num_ctx}
    _ollama_request("/api/generate", payload)


def build_ollama_prompt(context: str, query: str) -> str:
    """Serializa o prompt medido pelo MCKP e enviado ao Ollama."""
    return (
        "Baseado no contexto abaixo, responda a pergunta de forma direta e concisa.\n\n"
        f"CONTEXTO:\n{context}\n\nPERGUNTA: {query}\n\nRESPOSTA:"
    )


def create_ollama_strategy(
    model: str, *, truncate_context: bool = True
) -> Callable[[str, str], str]:
    """Estratégia para Ollama local via API nativa, com num_ctx explícito."""
    ollama_model = model.removeprefix("ollama/")
    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
    max_output_tokens = int(os.environ.get("OLLAMA_MAX_OUTPUT_TOKENS", "500"))
    # Aproximação conservadora para texto em inglês/português e espaço do template.
    char_limit = max(1_000, (num_ctx - max_output_tokens - 256) * 3)

    def strategy_fn(context: str, query: str) -> str:
        if truncate_context and len(context) > char_limit:
            logger.warning(
                f"[{model}] Contexto truncado: {len(context)} -> {char_limit} chars "
                f"(num_ctx={num_ctx})"
            )
            context = context[:char_limit]

        prompt = build_ollama_prompt(context, query)
        payload = {
            "model": ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": -1,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": max_output_tokens,
                "temperature": 0,
                "seed": 42,
            },
        }
        try:
            response = _ollama_request("/api/chat", payload)
            strategy_fn.last_call = {
                "sent_context": context,
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "total_duration_ns": response.get("total_duration"),
                "load_duration_ns": response.get("load_duration"),
            }
            return response.get("message", {}).get("content", "").strip()
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            logger.error(f"[{model}] Erro na API Ollama: {type(exc).__name__}: {exc}")
            strategy_fn.last_call = {"sent_context": context, "error": str(exc)}
            return f"[Erro: {exc}]"

    strategy_fn.last_call = {}
    return strategy_fn


def _route_api_strategy(model: str) -> Callable[[str, str], str]:
    """Valida o prefixo do modelo e retorna a estratégia Ollama correspondente."""
    if not model.startswith("ollama/"):
        raise ValueError(
            f"Modelo '{model}' não suportado. Use o prefixo 'ollama/' "
            f"(disponíveis: {', '.join(AVAILABLE_MODELS)})"
        )
    return create_ollama_strategy(model)


def _output_from_ollama_call(
    answer: str,
    base_fn: Callable[[str, str], str],
    fallback_context: str,
) -> StrategyOutput:
    """Registra o contexto efetivamente enviado, inclusive após truncamento."""
    call = dict(base_fn.last_call)
    sent_context = call.pop("sent_context", fallback_context)
    return StrategyOutput(
        answer=answer,
        compressed_context=sent_context,
        details={f"ollama_{key}": value for key, value in call.items()},
    )


def create_raw_strategy(model: str) -> Callable[[str, str], StrategyOutput]:
    """Estratégia baseline: envia contexto bruto para o LLM (compressão identidade)."""
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        answer = base_fn(context, query)
        return _output_from_ollama_call(answer, base_fn, context)

    return strategy_fn


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

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        chunks = slider.process(context, query)
        if len(chunks) <= max_chunks:
            selected = chunks
        else:
            # Distribui uniformemente: início, meio e fim
            indices = [int(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
            selected = [chunks[i] for i in indices]
        limited_context = "\n---\n".join(selected)
        answer = base_fn(limited_context, query)
        return StrategyOutput(answer=answer, compressed_context=limited_context)

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
    synthesis_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        chunks = slider.process(context, query)
        if len(chunks) <= max_chunks:
            selected = chunks
        else:
            indices = [int(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
            selected = [chunks[i] for i in indices]
        limited_context = "\n---\n".join(selected)

        def answer_chunk(chunk: str):
            worker = _route_api_strategy(model)
            answer = worker(chunk, query)
            return answer, dict(worker.last_call)

        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            partials = list(executor.map(answer_chunk, selected))
        candidates = "\n\n".join(
            f"CANDIDATO {index + 1}:\n{answer}"
            for index, (answer, _) in enumerate(partials)
        )
        synthesis_query = (
            f"Pergunta original: {query}\n"
            "Sintetize a resposta final usando somente os candidatos acima."
        )
        answer = synthesis_fn(candidates, synthesis_query)
        calls = [metadata for _, metadata in partials] + [dict(synthesis_fn.last_call)]
        details = {
            "parallel_num_chunks": len(selected),
            "parallel_prompt_eval_count": sum(
                int(call.get("prompt_eval_count") or 0) for call in calls
            ),
            "parallel_eval_count": sum(int(call.get("eval_count") or 0) for call in calls),
        }
        return StrategyOutput(
            answer=answer,
            compressed_context=limited_context,
            details=details,
        )

    return strategy_fn


def create_semantic_compression_strategy(
    model: str,
    compression_ratio: float = 0.4
) -> Callable[[str, str], str]:
    """
    Estratégia com compressão semântica via LLM.
    """
    compressor = OllamaSemanticCompressor(model_name=model)
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        compressed = compressor.compress(context, compression_ratio)
        answer = base_fn(compressed, query)
        return _output_from_ollama_call(answer, base_fn, compressed)

    return strategy_fn


def create_llmlingua_strategy(
    model: str,
    compression_ratio: float = 0.5,
) -> Callable[[str, str], StrategyOutput]:
    """Baseline de compressão LLMLingua v1 (scorer GPT-2, device cpu)."""
    from prompt_compression import LLMLinguaCompressor

    compressor = LLMLinguaCompressor(model_name="gpt2", device_map="cpu")
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        compressed = compressor.compress(context, compression_ratio)
        answer = base_fn(compressed, query)
        return _output_from_ollama_call(answer, base_fn, compressed)

    return strategy_fn


def create_llmlingua2_strategy(
    model: str,
    compression_ratio: float = 0.5,
) -> Callable[[str, str], StrategyOutput]:
    """Compressão LLMLingua-2 (encoder XLM-RoBERTa, device cpu)."""
    from prompt_compression import LLMLingua2Compressor

    compressor = LLMLingua2Compressor(device_map="cpu")
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        compressed = compressor.compress(context, compression_ratio)
        answer = base_fn(compressed, query)
        return _output_from_ollama_call(answer, base_fn, compressed)

    return strategy_fn


def create_selective_context_strategy(
    model: str,
    compression_ratio: float = 0.5,
) -> Callable[[str, str], StrategyOutput]:
    """Compressão Selective Context (auto-informação via GPT-2)."""
    from prompt_compression import SelectiveContextCompressor

    compressor = SelectiveContextCompressor(model_type="gpt2", lang="en")
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        compressed = compressor.compress(context, compression_ratio)
        answer = base_fn(compressed, query)
        return _output_from_ollama_call(answer, base_fn, compressed)

    return strategy_fn


def create_cpc_strategy(
    model: str,
    compression_ratio: float = 0.5,
) -> Callable[[str, str], StrategyOutput]:
    """Compressão CPC (aproximação por seleção de sentenças query-aware, MiniLM).

    A implementação oficial oferece checkpoints baseados em Mistral-7B e
    Llama-1B. Para evitar outro modelo concorrendo por memória com as LLMs
    locais, usa-se seleção extrativa por similaridade com a consulta, da mesma
    família sentence-level e declarada como aproximação.
    """
    from mckp.compressors import SentenceExtractiveCompressor

    compressor = SentenceExtractiveCompressor(embedding_model="all-MiniLM-L6-v2")
    base_fn = _route_api_strategy(model)

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        compressed = compressor.compress(context, compression_ratio, query)
        answer = base_fn(compressed, query)
        return _output_from_ollama_call(answer, base_fn, compressed)

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

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        chunks = rig.process(context, query)
        if not chunks:
            return StrategyOutput(answer=base_fn(context, query), compressed_context=context)

        combined_context = "\n---\n".join(chunks)
        answer = base_fn(combined_context, query)
        return StrategyOutput(answer=answer, compressed_context=combined_context)

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

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        chunks = rsaw.process(context, query)
        if not chunks:
            return StrategyOutput(answer=base_fn(context, query), compressed_context=context)
        compressed = chunks[0]
        answer = base_fn(compressed, query)
        return StrategyOutput(answer=answer, compressed_context=compressed)

    return strategy_fn


def create_mckp_strategy(
    model: str,
    *,
    mu: float | None = None,
    distance: str | None = None,
    budget_bucket: int | None = None,
) -> Callable[[str, str], str]:
    """
    Estratégia MCKP, compressão seletiva por partição resolvida como problema
    da mochila de múltipla escolha. Lê a configuração de mckp/config.json e
    ajusta o orçamento de contexto ao num_ctx da rodada.
    """
    config_path = Path(__file__).parent / "mckp" / "config.json"
    config = MCKPConfig.from_json(config_path)
    config.summarizer_model = model
    if mu is not None:
        config.mu = mu
    if distance is not None:
        config.distance = distance
    if budget_bucket is not None:
        config.budget_bucket = budget_bucket
    # O orçamento efetivo é derivado da janela de contexto da rodada.
    config.model_context_tokens = int(os.environ.get("OLLAMA_NUM_CTX", config.budget_tokens))
    config.output_tokens = int(os.environ.get("OLLAMA_MAX_OUTPUT_TOKENS", "500"))
    config.audit_log_path = os.environ.get("MCKP_AUDIT_LOG")
    config.validate()

    mckp = MCKPStrategy(config, prompt_builder=build_ollama_prompt)
    # O MCKP já valida o prompt final no mesmo contador usado pela otimização.
    base_fn = create_ollama_strategy(model, truncate_context=False)

    def strategy_fn(context: str, query: str) -> StrategyOutput:
        chunks = mckp.process(context, query)
        if not chunks:
            return StrategyOutput(answer=base_fn(context, query), compressed_context=context)
        compressed = chunks[0]
        answer = base_fn(compressed, query)
        diagnostics = mckp.last_diagnostics
        details = {
            f"mckp_{key}": value
            for key, value in diagnostics.items()
        }
        details["mckp_exact"] = int(config.budget_bucket == 1)
        call = dict(base_fn.last_call)
        call.pop("sent_context", None)
        details.update({f"ollama_{key}": value for key, value in call.items()})
        return StrategyOutput(
            answer=answer,
            compressed_context=compressed,
            details=details,
        )

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
    short = model.removeprefix("ollama/")

    # Simplifica nomes longos
    replacements = {
        "ollama/hf.co/mradermacher/Llama-4-Scout-17B-6E-Instruct-GGUF:Q4_K_S": "scout-17b-6e-q4ks",
        "ollama/gpt-oss:20b": "gpt-oss-20b-ollama",
        "ollama/llama3.1:8b-instruct-q8_0": "llama3.1-8b-instruct-q8",
        "ollama/gemma4:26b-mlx": "gemma4-26b-mlx",
        "ollama/qwen3.6:35b-mlx": "qwen3.6-35b-mlx",
        "ollama/llama3.1:8b-text-q4_K_M": "llama3.1-8b-text-q4km",
        "ollama/deepseek-r1:32b": "deepseek-r1-32b",
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
            f"Modelos Ollama a testar (comma-separated, prefixo 'ollama/'). "
            f"Disponíveis: {', '.join(AVAILABLE_MODELS)}."
        )
    )
    
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Estratégias a testar (comma-separated). Opções: raw, sliding_window, parallel_window, semantic_compression, llmlingua, llmlingua2, selective_context, cpc, rig, rsaw, mckp, mock, all"
    )
    
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help=(
            "Benchmarks a executar (comma-separated). Opções: needle_in_haystack, ruler, "
            "longbench, babilong, narrativeqa, qasper, infinitebench, zeroscrolls, "
            "naturalquestions, triviaqa, hotpotqa, musique, meeting_summarization, all"
        )
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "tests" / "ollama-local" / "adhoc"),
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

    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=8192,
        help="Janela de contexto enviada ao Ollama via num_ctx (padrão: 8192)",
    )
    parser.add_argument(
        "--mckp-mu",
        type=float,
        default=None,
        help="Sobrescreve a penalização de transição mu do MCKP.",
    )
    parser.add_argument(
        "--mckp-distance",
        choices=["compressor_family", "param_diff", "none"],
        default=None,
        help="Sobrescreve a função de distância do MCKP.",
    )
    parser.add_argument(
        "--mckp-budget-bucket",
        type=int,
        default=None,
        help="Quantização do orçamento; use 1 para solução exata.",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ollama_num_ctx <= 756:
        raise ValueError("--ollama-num-ctx deve ser maior que 756 tokens")
    os.environ["OLLAMA_NUM_CTX"] = str(args.ollama_num_ctx)
    os.environ["MCKP_AUDIT_LOG"] = str(Path(args.output_dir) / "mckp_audit.jsonl")
    
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
            "semantic_compression", "rig", "rsaw", "mckp"
        ]
    else:
        strategy_types = [s.strip() for s in args.strategies.split(",")]
    
    # Factory functions para cada tipo de estratégia
    strategy_factories = {
        "raw": lambda m: create_raw_strategy(m),
        "sliding_window": lambda m: create_sliding_window_strategy(m),
        "parallel_window": lambda m: create_parallel_window_strategy(m),
        "semantic_compression": lambda m: create_semantic_compression_strategy(m),
        "llmlingua": lambda m: create_llmlingua_strategy(m),
        "llmlingua2": lambda m: create_llmlingua2_strategy(m),
        "selective_context": lambda m: create_selective_context_strategy(m),
        "cpc": lambda m: create_cpc_strategy(m),
        "rig": lambda m: create_rig_strategy(m),
        "rsaw": lambda m: create_rsaw_strategy(m),
        "mckp": lambda m: create_mckp_strategy(
            m,
            mu=args.mckp_mu,
            distance=args.mckp_distance,
            budget_bucket=args.mckp_budget_bucket,
        ),
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
        print("ERRO: Nenhuma estratégia registrada. Verifique se o Ollama está rodando (http://localhost:11434) ou use --mock-only")
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
            "zeroscrolls": {"num_examples": 5},
            "naturalquestions": {"num_examples": 5},
            "triviaqa": {"num_examples": 5},
            "hotpotqa": {"num_examples": 5},
            "musique": {"num_examples": 5},
            "meeting_summarization": {"num_examples": 5},
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
            "zeroscrolls": {"num_examples": 20},
            "naturalquestions": {"num_examples": 20},
            "triviaqa": {"num_examples": 20},
            "hotpotqa": {"num_examples": 20},
            "musique": {"num_examples": 20},
            "meeting_summarization": {"num_examples": 20},
        }
    
    # Filtra benchmarks se especificado
    if args.benchmarks != "all":
        selected = [b.strip() for b in args.benchmarks.split(",")]
        benchmark_configs = {
            k: v for k, v in benchmark_configs.items()
            if k in selected
        }
    
    ollama_models = [m for m in models_to_test if m.startswith("ollama/")]

    # Executa benchmarks
    print(f"\nModelos: {models_to_test if models_to_test else ['mock']}")
    print(f"Estratégias base: {strategy_types}")
    print(f"Estratégias registradas: {list(runner.strategies.keys())}")
    print(f"Benchmarks: {list(benchmark_configs.keys())}")
    if ollama_models:
        print(f"Ollama num_ctx: {args.ollama_num_ctx}")
    print("-" * 70)

    try:
        for model in ollama_models:
            print(f"Carregando {model.removeprefix('ollama/')} (num_ctx={args.ollama_num_ctx})...")
            set_ollama_model_loaded(model, args.ollama_num_ctx, loaded=True)
        runner.run_all_benchmarks(benchmark_configs)
    finally:
        for model in ollama_models:
            try:
                print(f"Descarregando {model.removeprefix('ollama/')}...")
                set_ollama_model_loaded(model, args.ollama_num_ctx, loaded=False)
            except Exception as exc:
                logger.error(f"Não foi possível descarregar {model}: {exc}")
    
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
