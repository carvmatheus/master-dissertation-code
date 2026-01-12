"""
Benchmark Runner - Orquestrador para executar todos os benchmarks.

Executa as estratégias de contexto contra os benchmarks e
gera relatórios de comparação.
"""
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from .base import BaseBenchmark, BenchmarkResult
from .needle_haystack import NeedleInHaystackBenchmark
from .ruler import RulerBenchmark
from .longbench import LongBenchTasks


@dataclass
class StrategyConfig:
    """Configuração de uma estratégia para benchmark."""
    name: str
    fn: Callable[[str, str], str]  # (context, query) -> response
    description: str = ""


@dataclass
class BenchmarkReport:
    """Relatório consolidado de benchmark."""
    timestamp: str
    strategies: List[str]
    benchmarks: List[str]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """
    Orquestrador principal para executar benchmarks.
    
    Executa múltiplas estratégias contra múltiplos benchmarks
    e gera relatórios comparativos.
    """
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmarks disponíveis
        self.benchmarks: Dict[str, BaseBenchmark] = {
            "needle_in_haystack": NeedleInHaystackBenchmark(),
            "ruler": RulerBenchmark(),
            "longbench": LongBenchTasks(),
        }
        
        # Estratégias registradas
        self.strategies: Dict[str, StrategyConfig] = {}
        
        # Resultados
        self.results: List[BenchmarkResult] = []
    
    def register_strategy(
        self,
        name: str,
        fn: Callable[[str, str], str],
        description: str = ""
    ) -> None:
        """
        Registra uma estratégia para ser testada.
        
        Args:
            name: Nome identificador da estratégia
            fn: Função (context, query) -> response
            description: Descrição opcional
        """
        self.strategies[name] = StrategyConfig(
            name=name,
            fn=fn,
            description=description
        )
        print(f"Estratégia registrada: {name}")
    
    def run_benchmark(
        self,
        benchmark_name: str,
        strategy_name: Optional[str] = None,
        **benchmark_kwargs
    ) -> List[BenchmarkResult]:
        """
        Executa um benchmark específico.
        
        Args:
            benchmark_name: Nome do benchmark
            strategy_name: Nome da estratégia (ou todas se None)
            **benchmark_kwargs: Argumentos para generate_test_cases
            
        Returns:
            Lista de resultados
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark não encontrado: {benchmark_name}")
        
        benchmark = self.benchmarks[benchmark_name]
        strategies_to_run = (
            {strategy_name: self.strategies[strategy_name]}
            if strategy_name
            else self.strategies
        )
        
        results = []
        
        for strat_name, strat_config in strategies_to_run.items():
            print(f"\n>> Executando {benchmark_name} com estratégia: {strat_name}")
            
            strat_results = benchmark.run_all(
                strategy_fn=strat_config.fn,
                strategy_name=strat_name,
                **benchmark_kwargs
            )
            
            results.extend(strat_results)
            self.results.extend(strat_results)
            
            # Calcula média
            avg_score = sum(r.score for r in strat_results) / len(strat_results) if strat_results else 0
            avg_latency = sum(r.latency_ms for r in strat_results) / len(strat_results) if strat_results else 0
            
            print(f"   Score médio: {avg_score:.3f} | Latência média: {avg_latency:.1f}ms | {len(strat_results)} casos")
        
        return results
    
    def run_all_benchmarks(
        self,
        benchmark_configs: Optional[Dict[str, Dict]] = None
    ) -> List[BenchmarkResult]:
        """
        Executa todos os benchmarks registrados.
        
        Args:
            benchmark_configs: Dict de benchmark_name -> kwargs
            
        Returns:
            Lista completa de resultados
        """
        if benchmark_configs is None:
            benchmark_configs = {
                "needle_in_haystack": {"num_paragraphs": 20, "num_needles": 3},
                "ruler": {"context_sizes": [10, 25, 50]},
                "longbench": {"num_qa_cases": 5},
            }
        
        all_results = []
        
        for benchmark_name, kwargs in benchmark_configs.items():
            results = self.run_benchmark(benchmark_name, **kwargs)
            all_results.extend(results)
        
        return all_results
    
    def compute_summary(self) -> Dict[str, Any]:
        """
        Calcula sumário estatístico dos resultados.
        
        Returns:
            Dict com métricas agregadas
        """
        if not self.results:
            return {}
        
        # Agrupa por estratégia
        by_strategy = defaultdict(list)
        for r in self.results:
            by_strategy[r.strategy_name].append(r)
        
        # Agrupa por benchmark
        by_benchmark = defaultdict(list)
        for r in self.results:
            by_benchmark[r.benchmark_name].append(r)
        
        # Métricas por estratégia
        strategy_metrics = {}
        for strat, results in by_strategy.items():
            scores = [r.score for r in results]
            latencies = [r.latency_ms for r in results]
            
            strategy_metrics[strat] = {
                "avg_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "num_tests": len(results),
            }
        
        # Métricas por benchmark
        benchmark_metrics = {}
        for bench, results in by_benchmark.items():
            scores = [r.score for r in results]
            
            benchmark_metrics[bench] = {
                "avg_score": sum(scores) / len(scores),
                "num_tests": len(results),
            }
        
        # Matriz estratégia x benchmark
        matrix = {}
        for strat in by_strategy:
            matrix[strat] = {}
            for bench in by_benchmark:
                strat_bench_results = [
                    r for r in self.results
                    if r.strategy_name == strat and r.benchmark_name == bench
                ]
                if strat_bench_results:
                    matrix[strat][bench] = sum(r.score for r in strat_bench_results) / len(strat_bench_results)
                else:
                    matrix[strat][bench] = None
        
        return {
            "total_tests": len(self.results),
            "strategies": strategy_metrics,
            "benchmarks": benchmark_metrics,
            "strategy_x_benchmark_matrix": matrix,
        }
    
    def generate_report(self) -> BenchmarkReport:
        """
        Gera relatório completo.
        
        Returns:
            BenchmarkReport com todos os dados
        """
        summary = self.compute_summary()
        
        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            strategies=list(self.strategies.keys()),
            benchmarks=list(self.benchmarks.keys()),
            results=[r.to_dict() for r in self.results],
            summary=summary,
        )
    
    def save_results_csv(self, filename: str = "benchmark_results.csv") -> str:
        """
        Salva resultados em CSV.
        
        Returns:
            Caminho do arquivo salvo
        """
        filepath = self.output_dir / filename
        
        if not self.results:
            print("Nenhum resultado para salvar.")
            return str(filepath)
        
        # Colunas principais
        fieldnames = [
            "benchmark", "strategy", "test_case", "score", "latency_ms"
        ]
        
        # Adiciona colunas de metadados comuns
        all_detail_keys = set()
        for r in self.results:
            all_detail_keys.update(r.details.keys())
        
        # Remove campos muito longos
        excluded_keys = {"response", "needle_fact"}
        detail_keys = sorted(all_detail_keys - excluded_keys)
        fieldnames.extend(detail_keys)
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            
            for r in self.results:
                row = r.to_dict()
                # Flatten details
                for k in detail_keys:
                    row[k] = r.details.get(k, "")
                writer.writerow(row)
        
        print(f"Resultados salvos em: {filepath}")
        return str(filepath)
    
    def save_results_json(self, filename: str = "benchmark_results.json") -> str:
        """
        Salva resultados e sumário em JSON.
        
        Returns:
            Caminho do arquivo salvo
        """
        filepath = self.output_dir / filename
        
        report = self.generate_report()
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"Relatório JSON salvo em: {filepath}")
        return str(filepath)
    
    def save_comparison_table(self, filename: str = "comparison_table.csv") -> str:
        """
        Salva tabela de comparação estratégia x benchmark.
        
        Returns:
            Caminho do arquivo salvo
        """
        filepath = self.output_dir / filename
        summary = self.compute_summary()
        
        matrix = summary.get("strategy_x_benchmark_matrix", {})
        
        if not matrix:
            print("Nenhum dado para tabela de comparação.")
            return str(filepath)
        
        benchmarks = list(self.benchmarks.keys())
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Strategy"] + benchmarks + ["Average"])
            
            # Rows
            for strat, scores in matrix.items():
                row = [strat]
                bench_scores = []
                for bench in benchmarks:
                    score = scores.get(bench)
                    if score is not None:
                        row.append(f"{score:.3f}")
                        bench_scores.append(score)
                    else:
                        row.append("-")
                
                # Average
                if bench_scores:
                    avg = sum(bench_scores) / len(bench_scores)
                    row.append(f"{avg:.3f}")
                else:
                    row.append("-")
                
                writer.writerow(row)
        
        print(f"Tabela de comparação salva em: {filepath}")
        return str(filepath)
    
    def print_summary(self) -> None:
        """Imprime sumário no console."""
        summary = self.compute_summary()
        
        print("\n" + "=" * 60)
        print("SUMÁRIO DOS BENCHMARKS")
        print("=" * 60)
        
        print(f"\nTotal de testes: {summary.get('total_tests', 0)}")
        
        print("\n--- Métricas por Estratégia ---")
        for strat, metrics in summary.get("strategies", {}).items():
            print(f"\n{strat}:")
            print(f"  Score médio: {metrics['avg_score']:.3f}")
            print(f"  Score min/max: {metrics['min_score']:.3f} / {metrics['max_score']:.3f}")
            print(f"  Latência média: {metrics['avg_latency_ms']:.1f}ms")
        
        print("\n--- Matriz Estratégia x Benchmark ---")
        matrix = summary.get("strategy_x_benchmark_matrix", {})
        
        if matrix:
            benchmarks = list(next(iter(matrix.values())).keys()) if matrix else []
            
            # Header
            header = f"{'Strategy':<20}" + "".join(f"{b:<20}" for b in benchmarks)
            print(header)
            print("-" * len(header))
            
            for strat, scores in matrix.items():
                row = f"{strat:<20}"
                for bench in benchmarks:
                    score = scores.get(bench)
                    if score is not None:
                        row += f"{score:.3f}".ljust(20)
                    else:
                        row += "-".ljust(20)
                print(row)
        
        print("\n" + "=" * 60)
