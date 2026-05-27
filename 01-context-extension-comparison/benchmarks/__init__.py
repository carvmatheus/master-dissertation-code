"""
Benchmarks para avaliação de estratégias de extensão de contexto.

Implementa testes especializados baseados em:
- Needle-in-a-Haystack: recuperação de informação em posição aleatória
- RULER: medição do "tamanho real de contexto"
- LongBench: tarefas multitarefa para contexto longo
- BABILong: raciocínio sobre fatos dispersos em contexto longo (Kuratov et al., 2024)
- NarrativeQA: QA sobre livros e roteiros completos (Kočiský et al., 2018)
- QASPER: QA sobre artigos científicos completos (Dasigi et al., 2021)
- InfiniteBench: QA sobre contextos >100K tokens (Zhang et al., 2024)

Referências:
- LongBench: Bai et al., 2023
- RULER: Hsieh et al., 2024
"""

from .needle_haystack import NeedleInHaystackBenchmark
from .ruler import RulerBenchmark
from .longbench import LongBenchTasks
from .babilong import BABILongBenchmark
from .narrativeqa import NarrativeQABenchmark
from .qasper import QASPERBenchmark
from .infinitebench import InfiniteBenchBenchmark
from .runner import BenchmarkRunner, BenchmarkResult

__all__ = [
    "NeedleInHaystackBenchmark",
    "RulerBenchmark",
    "LongBenchTasks",
    "BABILongBenchmark",
    "NarrativeQABenchmark",
    "QASPERBenchmark",
    "InfiniteBenchBenchmark",
    "BenchmarkRunner",
    "BenchmarkResult",
]
