"""
Validadores de compressão: CUSTO (tamanho) e QUALIDADE (preservação).

Separam os dois eixos que a performance downstream sozinha não revela — os
mesmos `L_{j,o}` (custo) e `Q_{j,o}` (qualidade) da formulação knapsack.
"""
from .cost import CostMetrics, count_tokens, measure_cost
from .quality import QualityMetrics, measure_quality

__all__ = [
    "CostMetrics",
    "QualityMetrics",
    "count_tokens",
    "measure_cost",
    "measure_quality",
]
