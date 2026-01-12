"""
Classes base para os benchmarks.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import time


@dataclass
class BenchmarkResult:
    """Resultado de um teste de benchmark."""
    benchmark_name: str
    strategy_name: str
    test_case: str
    score: float  # 0.0 a 1.0
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "strategy": self.strategy_name,
            "test_case": self.test_case,
            "score": self.score,
            "latency_ms": self.latency_ms,
            **self.details
        }


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
            response = strategy_fn(test_case.context, test_case.query)
            latency_ms = (time.perf_counter() - start) * 1000
            
            score = self.evaluate_response(response, test_case.expected, test_case)
            
            return BenchmarkResult(
                benchmark_name=self.name,
                strategy_name=strategy_name,
                test_case=test_case.name,
                score=score,
                latency_ms=latency_ms,
                details={
                    "response": response[:500] if response else "",
                    "expected": test_case.expected,
                    **test_case.metadata
                }
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
        
        for tc in test_cases:
            result = self.run_single(tc, strategy_fn, strategy_name)
            results.append(result)
        
        return results
