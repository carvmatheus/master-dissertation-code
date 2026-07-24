"""
Reconstrução do contexto comprimido.

Remonta, na ordem original das partições, os textos das opções escolhidas pelo
solver. Partições omitidas não geram texto: isso evita introduzir marcadores cujo
custo não tenha sido considerado pela otimização. Cada opção não vazia após a
primeira partição inclui conservadoramente o custo de um separador.
"""
from __future__ import annotations

from typing import List

from .models import CompressionOption
from .tokenization import TokenCounter

_SEP = "\n---\n"


def serialized_option_cost(text: str, partition_index: int, counter: TokenCounter) -> int:
    """Custo conservador da opção no contexto reconstruído.

    Uma opção posterior à partição inicial reserva um separador mesmo quando
    todas as partições anteriores forem omitidas. Nesse caso o custo real fica
    menor, nunca maior, que o custo usado pelo solver.
    """
    if not text:
        return 0
    prefix = _SEP if partition_index > 0 else ""
    return counter.count(prefix + text)


def reconstruct(chosen: List[CompressionOption]) -> str:
    """Concatena apenas opções não omitidas, preservando a ordem original."""
    ordered = sorted(chosen, key=lambda o: o.partition_index)
    parts = [opt.text for opt in ordered if opt.text]
    return _SEP.join(parts)
