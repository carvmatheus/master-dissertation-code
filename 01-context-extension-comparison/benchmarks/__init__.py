"""
Benchmarks para avaliação de estratégias de extensão de contexto.

Implementa testes especializados baseados em:
- Needle-in-a-Haystack: recuperação de informação em posição aleatória
- RULER: medição do "tamanho real de contexto" 
- LongBench: tarefas multitarefa para contexto longo

Referências:
- LongBench: Bai et al., 2023
- RULER: Hsieh et al., 2024
"""

from .needle_haystack import NeedleInHaystackBenchmark
from .ruler import RulerBenchmark
from .longbench import LongBenchTasks
from .runner import BenchmarkRunner, BenchmarkResult

__all__ = [
    "NeedleInHaystackBenchmark",
    "RulerBenchmark", 
    "LongBenchTasks",
    "BenchmarkRunner",
    "BenchmarkResult",
]
