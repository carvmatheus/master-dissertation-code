"""
Validador de CUSTO da compressão.

Mede o "tamanho" da representação comprimida em relação ao original — o eixo
`L_{j,o}` (comprimento) da formulação knapsack (PIBPC) e o "quanto foi reduzido"
da teoria de densificação informacional.

Todas as métricas são determinísticas e locais (tiktoken), sem custo de API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

_ENC = None


def _enc():
    """Tokenizer compartilhado — mesmo encoding usado pelo RSAW (o200k_base)."""
    global _ENC
    if _ENC is None:
        import tiktoken

        try:
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:
            _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc().encode(text))


@dataclass
class CostMetrics:
    """Métricas de custo de uma compressão original -> comprimido."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # comprimido/original em (0, 1]; menor = mais comprimido
    tokens_saved: int
    char_ratio: float
    budget_tokens: Optional[int] = None
    fits_budget: Optional[bool] = None

    def as_details(self) -> Dict[str, object]:
        d: Dict[str, object] = {
            "cost_original_tokens": self.original_tokens,
            "cost_compressed_tokens": self.compressed_tokens,
            "cost_compression_ratio": round(self.compression_ratio, 4),
            "cost_tokens_saved": self.tokens_saved,
            "cost_char_ratio": round(self.char_ratio, 4),
        }
        if self.budget_tokens is not None:
            d["cost_budget_tokens"] = self.budget_tokens
            d["cost_fits_budget"] = int(bool(self.fits_budget))
        return d


def measure_cost(
    original: str,
    compressed: str,
    budget_tokens: Optional[int] = None,
) -> CostMetrics:
    """Compara o tamanho do contexto comprimido com o original.

    Args:
        original: contexto original (antes da compressão).
        compressed: contexto efetivamente enviado ao LLM pela estratégia.
        budget_tokens: orçamento B, se aplicável (para budget satisfaction).
    """
    ot = count_tokens(original)
    ct = count_tokens(compressed)
    ratio = (ct / ot) if ot else 1.0
    char_ratio = (len(compressed) / len(original)) if original else 1.0
    fits = (ct <= budget_tokens) if budget_tokens is not None else None
    return CostMetrics(
        original_tokens=ot,
        compressed_tokens=ct,
        compression_ratio=ratio,
        tokens_saved=ot - ct,
        char_ratio=char_ratio,
        budget_tokens=budget_tokens,
        fits_budget=fits,
    )
