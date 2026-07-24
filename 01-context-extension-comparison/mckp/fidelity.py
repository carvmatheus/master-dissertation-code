"""
Fidelidade f_{j,o} de uma opção de compressão.

Embrulho fino sobre os validadores reference-free já existentes
(validators/quality.py), para que o MCKP e a suíte de benchmarks usem
exatamente a mesma definição de fidelidade e retenção factual. Nada é
recalculado aqui, apenas roteado com tratamento de casos de borda.
"""
from __future__ import annotations

from typing import Optional

from .tokenization import DEFAULT_TOKEN_COUNTER

try:
    from validators.quality import semantic_fidelity as _semantic_fidelity
    from validators.quality import number_retention as _number_retention
except ImportError:  # execução como pacote
    from ..validators.quality import semantic_fidelity as _semantic_fidelity
    from ..validators.quality import number_retention as _number_retention


def count_tokens(text: str) -> int:
    """c_{j,o}, contagem de tokens da representação (encoding o200k_base)."""
    return DEFAULT_TOKEN_COUNTER.count(text)


def fidelity(original: str, compressed: str, enable_semantic: bool = True) -> float:
    """f_{j,o} em [0, 1], cobertura semântica do original pelo comprimido.

    Casos de borda. Comprimido vazio (omissão) tem fidelidade 0. Texto
    idêntico tem fidelidade 1. Quando o validador não consegue medir (texto
    muito curto), assume-se preservação total apenas se o texto não mudou.
    """
    if not compressed:
        return 0.0
    if compressed == original:
        return 1.0
    if not enable_semantic:
        try:
            from validators.quality import content_recall
        except ImportError:  # execução como pacote
            from ..validators.quality import content_recall
        lexical = content_recall(original, compressed)
        return 0.5 if lexical is None else max(0.0, min(1.0, float(lexical)))

    value = _semantic_fidelity(original, compressed)
    if value is None:
        return 1.0 if compressed.strip() == original.strip() else 0.5
    return max(0.0, min(1.0, float(value)))


def number_retention(original: str, compressed: str) -> Optional[float]:
    """Fração de números, datas e valores do original preservados.

    Retorna None quando a partição não contém itens factuais, caso em que o
    pré-filtro de retenção não se aplica.
    """
    return _number_retention(original, compressed)
