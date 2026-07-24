"""
Classes base para os benchmarks.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
import os
import time


@dataclass
class StrategyOutput:
    """Saída rica de uma estratégia: resposta do LLM + o que foi enviado a ele.

    Estratégias que comprimem o contexto devem retornar isto (em vez de apenas
    a string de resposta) para que os validadores de custo/qualidade possam
    comparar o `compressed_context` com o contexto original.
    """
    answer: str
    compressed_context: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Resultado de um teste de benchmark."""
    benchmark_name: str
    strategy_name: str
    test_case: str
    score: float  # 0.0 a 1.0
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    executed_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "strategy": self.strategy_name,
            "test_case": self.test_case,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "executed_at": self.executed_at,
            **self.details
        }


def _compute_validators(
    original: str,
    compressed: str,
    expected: str,
) -> Dict[str, Any]:
    """Roda os validadores de custo e qualidade de forma tolerante a falhas.

    Semântico (Tier 2) é ligado por padrão; desligue com VALIDATOR_SEMANTIC=0
    para runs mais rápidos. Qualquer erro degrada para {} sem interromper o run.
    """
    try:
        from validators import measure_cost, measure_quality
    except Exception:
        return {}

    details: Dict[str, Any] = {}
    try:
        cost = measure_cost(original, compressed)
        details.update(cost.as_details())
        enable_semantic = os.environ.get("VALIDATOR_SEMANTIC", "1") != "0"
        quality = measure_quality(
            original,
            compressed,
            expected=expected or "",
            compression_ratio=cost.compression_ratio,
            enable_semantic=enable_semantic,
        )
        details.update(quality.as_details())
    except Exception as exc:  # nunca deixa o validador derrubar o benchmark
        details["validator_error"] = f"{type(exc).__name__}: {exc}"
    return details


@dataclass
class TestCase:
    """Um caso de teste individual."""
    name: str
    context: str  # Texto de contexto (haystack)
    query: str    # Pergunta
    expected: str # Resposta esperada
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    """Classe base para benchmarks."""
    
    name: str = "base"
    
    @abstractmethod
    def generate_test_cases(self, **kwargs) -> List[TestCase]:
        """Gera casos de teste para este benchmark."""
        pass
    
    @abstractmethod
    def evaluate_response(
        self, 
        response: str, 
        expected: str,
        test_case: TestCase
    ) -> float:
        """
        Avalia uma resposta contra o esperado.
        
        Returns:
            Score de 0.0 a 1.0
        """
        pass
    
    def run_single(
        self,
        test_case: TestCase,
        strategy_fn: Callable[[str, str], str],
        strategy_name: str,
    ) -> BenchmarkResult:
        """
        Executa um único caso de teste.
        
        Args:
            test_case: Caso de teste
            strategy_fn: Função que recebe (context, query) e retorna resposta
            strategy_name: Nome da estratégia sendo testada
            
        Returns:
            Resultado do benchmark
        """
        start = time.perf_counter()

        try:
            output = strategy_fn(test_case.context, test_case.query)
            latency_ms = (time.perf_counter() - start) * 1000

            # Aceita string simples (retrocompatível) ou StrategyOutput rico.
            if isinstance(output, StrategyOutput):
                response = output.answer
                compressed_context = output.compressed_context
                strategy_details = output.details
            else:
                response = output
                compressed_context = None
                strategy_details = {}

            score = self.evaluate_response(response, test_case.expected, test_case)

            details: Dict[str, Any] = {
                "response": response[:500] if response else "",
                "expected": test_case.expected,
                **test_case.metadata,
                **strategy_details,
            }

            # Validadores de custo/qualidade sobre o que a estratégia comprimiu.
            if compressed_context is not None:
                details.update(
                    _compute_validators(test_case.context, compressed_context, test_case.expected)
                )

            return BenchmarkResult(
                benchmark_name=self.name,
                strategy_name=strategy_name,
                test_case=test_case.name,
                score=score,
                latency_ms=latency_ms,
                details=details,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return BenchmarkResult(
                benchmark_name=self.name,
                strategy_name=strategy_name,
                test_case=test_case.name,
                score=0.0,
                latency_ms=latency_ms,
                details={"error": str(e)}
            )
    
    def run_all(
        self,
        strategy_fn: Callable[[str, str], str],
        strategy_name: str,
        **generate_kwargs
    ) -> List[BenchmarkResult]:
        """
        Executa todos os casos de teste para uma estratégia.
        
        Returns:
            Lista de resultados
        """
        test_cases = self.generate_test_cases(**generate_kwargs)
        results = []
        total = len(test_cases)

        for idx, tc in enumerate(test_cases, start=1):
            result = self.run_single(tc, strategy_fn, strategy_name)
            results.append(result)
            err = result.details.get("error") if isinstance(result.details, dict) else None
            status = f"ERRO: {err}" if err else f"score {result.score:.2f}"
            print(
                f"     [{idx}/{total}] {tc.name}: {status} | {result.latency_ms:.0f}ms",
                flush=True,
            )

        return results
