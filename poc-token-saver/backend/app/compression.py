import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

# Adiciona o diretório da dissertação ao path
REPO_ROOT = Path(__file__).resolve().parents[3]
DISSERTATION_DIR = REPO_ROOT / "01-context-extension-comparison"
if str(DISSERTATION_DIR) not in sys.path:
    sys.path.insert(0, str(DISSERTATION_DIR))

from mckp.strategy import MCKPStrategy
from mckp.models import MCKPConfig
from context_strategies import SlidingWindowStrategy
import tiktoken

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "cl100k_base"


def _count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding(DEFAULT_TOKENIZER)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _build_mckp_strategy(budget: int) -> MCKPStrategy:
    """Constroi uma instância leve de MCKPStrategy para a PoC."""
    config = MCKPConfig(
        partitioner="structural",
        max_partition_tokens=300,
        budget_tokens=budget,
        model_context_tokens=None,
        output_tokens=500,
        token_safety_margin=128,
        budget_bucket=1,
        option_set=[
            {"compressor": "identity", "param": 1.0},
            {"compressor": "sentence_extractive", "param": 0.6},
            {"compressor": "sentence_extractive", "param": 0.3},
            {"compressor": "omission", "param": 0.0},
        ],
        audit_log_path=None,
        audit_include_text=False,
    )
    return MCKPStrategy(config=config)


def compress_document(
    document: str,
    query: str,
    strategy: str = "MCKP",
    budget: int = 4000,
) -> Dict:
    """
    Comprime um documento usando a estratégia escolhida.

    Retorna dict com métricas e o contexto comprimido.
    """
    start = time.perf_counter()

    original_tokens = _count_tokens(document)

    if strategy.upper() == "MCKP":
        compressor = _build_mckp_strategy(budget)
        result_list = compressor.process(document, query)
        compressed = result_list[0] if result_list else ""
        diagnostics = compressor.last_diagnostics
    elif strategy.upper() == "SLIDING_WINDOW":
        sw = SlidingWindowStrategy(chunk_size=budget, overlap=200)
        chunks = sw.process(document, query)
        compressed = "\n---\n".join(chunks[:3])  # limita a 3 janelas para a PoC
        diagnostics = {"num_chunks": len(chunks)}
    else:
        raise ValueError(f"Estratégia não suportada: {strategy}")

    final_tokens = _count_tokens(compressed)
    execution_time_ms = round((time.perf_counter() - start) * 1000, 2)

    savings_percentage = 0.0
    if original_tokens > 0:
        savings_percentage = round(
            (original_tokens - final_tokens) / original_tokens * 100, 2
        )

    return {
        "compressed_context": compressed,
        "original_tokens": original_tokens,
        "final_tokens": final_tokens,
        "savings_percentage": savings_percentage,
        "execution_time_ms": execution_time_ms,
        "strategy": strategy.upper(),
        "budget": budget,
        "diagnostics": diagnostics,
    }
