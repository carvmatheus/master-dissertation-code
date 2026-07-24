"""Política única de contagem de tokens usada pelo pipeline MCKP."""
from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    """Contrato mínimo para substituir o tokenizer sem alterar o pipeline."""

    def count(self, text: str) -> int:
        ...


class TiktokenTokenCounter:
    """Contador compatível com os validadores dos benchmarks."""

    def count(self, text: str) -> int:
        try:
            from validators.cost import count_tokens
        except ImportError:  # execução como pacote
            from ..validators.cost import count_tokens
        return count_tokens(text)


DEFAULT_TOKEN_COUNTER = TiktokenTokenCounter()
