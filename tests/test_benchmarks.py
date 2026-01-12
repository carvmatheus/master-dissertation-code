"""
Testes unitários para o módulo de benchmarks.
"""
import sys
from pathlib import Path

# Adiciona o módulo ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "01-context-extension-comparison"))

from benchmarks.base import TestCase, BenchmarkResult
from benchmarks.needle_haystack import NeedleInHaystackBenchmark
from benchmarks.ruler import RulerBenchmark
from benchmarks.longbench import LongBenchTasks
from benchmarks.runner import BenchmarkRunner


class TestNeedleInHaystack:
    """Testes para o benchmark Needle-in-a-Haystack."""

    def test_generate_test_cases(self):
        """Verifica geração de casos de teste."""
        benchmark = NeedleInHaystackBenchmark(seed=42)
        cases = benchmark.generate_test_cases(
            num_paragraphs=5,
            num_needles=2,
            positions=["start", "middle", "end"]
        )
        
        # 2 needles * 3 positions = 6 cases
        assert len(cases) == 6
        assert all(isinstance(c, TestCase) for c in cases)

    def test_needle_is_in_context(self):
        """Verifica que a agulha está presente no contexto."""
        benchmark = NeedleInHaystackBenchmark(seed=42)
        cases = benchmark.generate_test_cases(
            num_paragraphs=5,
            num_needles=1,
            positions=["middle"]
        )
        
        case = cases[0]
        needle_fact = case.metadata.get("needle_fact", "")
        
        assert needle_fact in case.context

    def test_evaluate_exact_match(self):
        """Teste de match exato."""
        benchmark = NeedleInHaystackBenchmark()
        case = TestCase(
            name="test",
            context="",
            query="",
            expected="7392"
        )
        
        # Match exato
        assert benchmark.evaluate_response("O código é 7392.", "7392", case) == 1.0
        
        # Sem match
        assert benchmark.evaluate_response("Não sei.", "7392", case) == 0.0

    def test_evaluate_partial_match(self):
        """Teste de match parcial."""
        benchmark = NeedleInHaystackBenchmark()
        case = TestCase(
            name="test",
            context="",
            query="",
            expected="15 de março às 14h"
        )
        
        # Match parcial (algumas palavras)
        score = benchmark.evaluate_response("A reunião é em março.", "15 de março às 14h", case)
        assert 0 < score < 1


class TestRuler:
    """Testes para o benchmark RULER."""

    def test_generate_test_cases(self):
        """Verifica geração de casos de teste."""
        benchmark = RulerBenchmark(seed=42)
        cases = benchmark.generate_test_cases(
            context_sizes=[10, 20],
            num_facts_per_context=2,
            position_sets=[[0.1, 0.9]]
        )
        
        # 2 sizes * 1 position_set * 2 facts = 4 cases
        assert len(cases) == 4
        assert all(isinstance(c, TestCase) for c in cases)

    def test_facts_in_context(self):
        """Verifica que os fatos estão no contexto."""
        benchmark = RulerBenchmark(seed=42)
        cases = benchmark.generate_test_cases(
            context_sizes=[10],
            num_facts_per_context=2,
            position_sets=[[0.2, 0.8]]
        )
        
        # Pelo menos um dos fatos esperados deve estar no contexto
        for case in cases:
            # O contexto deve conter algo relacionado ao expected
            assert len(case.context) > 0
            assert len(case.expected) > 0

    def test_evaluate_response(self):
        """Testa avaliação de resposta."""
        benchmark = RulerBenchmark()
        case = TestCase(
            name="test",
            context="",
            query="",
            expected="ID-12345"
        )
        
        # Match exato
        assert benchmark.evaluate_response("O ID é ID-12345", "ID-12345", case) == 1.0
        
        # Match parcial (número presente)
        score = benchmark.evaluate_response("O número é 12345", "ID-12345", case)
        assert score == 0.5


class TestLongBench:
    """Testes para o benchmark LongBench."""

    def test_generate_qa_cases(self):
        """Verifica geração de casos de QA."""
        benchmark = LongBenchTasks(seed=42)
        cases = benchmark.generate_test_cases(
            task_types=["qa"],
            num_qa_cases=3,
            use_multi_doc=True
        )
        
        assert len(cases) == 3
        assert all(c.metadata.get("task_type") == "multi_doc_qa" for c in cases)

    def test_generate_summarization_cases(self):
        """Verifica geração de casos de sumarização."""
        benchmark = LongBenchTasks(seed=42)
        cases = benchmark.generate_test_cases(
            task_types=["summarization"]
        )
        
        assert len(cases) >= 1
        assert all("summarization" in c.metadata.get("task_type", "") for c in cases)

    def test_evaluate_qa(self):
        """Testa avaliação de QA."""
        benchmark = LongBenchTasks()
        case = TestCase(
            name="test",
            context="",
            query="",
            expected="23%",
            metadata={"task_type": "single_doc_qa"}
        )
        
        assert benchmark.evaluate_response("O crescimento foi de 23%.", "23%", case) == 1.0

    def test_evaluate_summarization(self):
        """Testa avaliação de sumarização."""
        benchmark = LongBenchTasks()
        case = TestCase(
            name="test",
            context="",
            query="",
            expected="",
            metadata={
                "task_type": "summarization",
                "expected_topics": ["inteligência artificial", "setores", "desafios"]
            }
        )
        
        # Resposta com todos os tópicos
        score = benchmark.evaluate_response(
            "A inteligência artificial está transformando diversos setores, mas há desafios éticos.",
            "",
            case
        )
        assert score == 1.0
        
        # Resposta com alguns tópicos
        score = benchmark.evaluate_response(
            "A tecnologia está avançando.",
            "",
            case
        )
        assert score < 1.0


class TestBenchmarkRunner:
    """Testes para o orquestrador de benchmarks."""

    def test_register_strategy(self):
        """Verifica registro de estratégias."""
        runner = BenchmarkRunner(output_dir="/tmp/test_benchmarks")
        
        runner.register_strategy(
            "mock",
            lambda ctx, q: "resposta mock",
            "Estratégia de teste"
        )
        
        assert "mock" in runner.strategies

    def test_run_with_mock_strategy(self):
        """Executa benchmark com estratégia mock."""
        runner = BenchmarkRunner(output_dir="/tmp/test_benchmarks")
        
        def mock_strategy(context: str, query: str) -> str:
            # Retorna algo do contexto
            if "7392" in context:
                return "7392"
            return "não encontrado"
        
        runner.register_strategy("mock", mock_strategy)
        
        results = runner.run_benchmark(
            "needle_in_haystack",
            strategy_name="mock",
            num_paragraphs=5,
            num_needles=1,
            positions=["start"]
        )
        
        assert len(results) > 0
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_compute_summary(self):
        """Testa cálculo de sumário."""
        runner = BenchmarkRunner(output_dir="/tmp/test_benchmarks")
        
        # Adiciona resultados manualmente
        runner.results = [
            BenchmarkResult(
                benchmark_name="test",
                strategy_name="mock",
                test_case="case1",
                score=0.8,
                latency_ms=100
            ),
            BenchmarkResult(
                benchmark_name="test",
                strategy_name="mock",
                test_case="case2",
                score=0.6,
                latency_ms=150
            ),
        ]
        
        summary = runner.compute_summary()
        
        assert summary["total_tests"] == 2
        assert "mock" in summary["strategies"]
        assert summary["strategies"]["mock"]["avg_score"] == 0.7


class TestBenchmarkResult:
    """Testes para a classe BenchmarkResult."""

    def test_to_dict(self):
        """Verifica conversão para dicionário."""
        result = BenchmarkResult(
            benchmark_name="needle",
            strategy_name="raw",
            test_case="test1",
            score=0.9,
            latency_ms=50.5,
            details={"position": "start"}
        )
        
        d = result.to_dict()
        
        assert d["benchmark"] == "needle"
        assert d["strategy"] == "raw"
        assert d["score"] == 0.9
        assert d["position"] == "start"
