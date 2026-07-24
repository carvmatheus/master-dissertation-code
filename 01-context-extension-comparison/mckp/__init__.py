"""
Framework de Compressão Seletiva de Contexto como MCKP.

Reimplementação do método como um problema da mochila de múltipla escolha exato,
resolvido por programação dinâmica com penalização de transição. Substitui a
heurística gulosa do baseline RSAW, que é mantido para comparação.
"""
from .models import CompressionOption, MCKPConfig, Partition, Solution
from .solver import InfeasibleMCKPError, MCKPSolver
from .strategy import MCKPBudgetError, MCKPStrategy
from .tokenization import TiktokenTokenCounter, TokenCounter

__all__ = [
    "MCKPStrategy",
    "MCKPConfig",
    "MCKPSolver",
    "InfeasibleMCKPError",
    "MCKPBudgetError",
    "TokenCounter",
    "TiktokenTokenCounter",
    "Partition",
    "CompressionOption",
    "Solution",
]
