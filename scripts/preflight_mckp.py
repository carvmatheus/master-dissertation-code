#!/usr/bin/env python3
"""Valida dependências, datasets e compressores antes de uma rodada MCKP."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "01-context-extension-comparison"
DATA_DIR = REPO_ROOT / "data" / "benchmarks"
sys.path.insert(0, str(PROJECT_DIR))


EXPECTED_VERSIONS = {
    "llmlingua": "0.2.2",
    "transformers": "4.46.3",
    "selective-context": "0.1.4",
}


def validate_versions() -> None:
    errors = []
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} não está instalado")
            continue
        if actual != expected:
            errors.append(f"{package}={actual}; esperado {expected}")
    if errors:
        raise RuntimeError("dependências incompatíveis: " + "; ".join(errors))


def validate_jsonl(name: str, minimum: int) -> int:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"dataset ausente: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < minimum:
        raise RuntimeError(f"{name} contém {len(rows)} casos; mínimo {minimum}")
    for index, row in enumerate(rows):
        if not row.get("context") or not row.get("query") or not row.get("answers"):
            raise RuntimeError(f"{name}[{index}] não segue o schema context/query/answers")
    if name == "naturalquestions":
        sources = {row.get("metadata", {}).get("source") for row in rows}
        if sources != {"google-research-datasets/natural_questions"}:
            raise RuntimeError("Natural Questions ainda usa o adaptador QA-pair inválido")
    if name == "longbench":
        sources = {row.get("metadata", {}).get("source") for row in rows}
        if sources != {"zai-org/LongBench"}:
            raise RuntimeError("LongBench não contém amostras oficiais")
    return len(rows)


def validate_compressors() -> dict:
    from mckp import MCKPConfig, MCKPStrategy

    config = MCKPConfig.from_json(PROJECT_DIR / "mckp" / "config.json")
    config.budget_tokens = 70
    config.model_context_tokens = None
    config.audit_log_path = None
    text = (
        "Paris is the capital of France. It is known for the Eiffel Tower. "
        "Brasilia is the capital of Brazil. Brazil is in South America. "
    ) * 4
    strategy = MCKPStrategy(config)
    strategy.process(text, "What is the capital of France?")
    diagnostics = strategy.last_diagnostics
    if diagnostics["num_compressor_failures"]:
        raise RuntimeError(f"falhas no preflight: {diagnostics['compressor_failures']}")
    evaluated = {
        option["compressor"]
        for partition in diagnostics["audit_record"]["partitions"]
        for option in partition["options"]
    }
    required = set(config.required_compressors)
    if not required.issubset(evaluated):
        raise RuntimeError(f"opções ausentes: {sorted(required - evaluated)}")
    return {
        "required_compressors": sorted(required),
        "evaluated_compressors": sorted(evaluated),
        "num_options": diagnostics["num_options"],
    }


def validate_runner() -> None:
    from benchmarks.real_world import LongBenchBenchmark
    from benchmarks.runner import BenchmarkRunner

    benchmark = BenchmarkRunner(output_dir="/tmp/mckp-preflight-runner").benchmarks[
        "longbench"
    ]
    if not isinstance(benchmark, LongBenchBenchmark):
        raise RuntimeError("runner ainda aponta para o LongBench sintético")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        default=(
            "longbench,zeroscrolls,naturalquestions,triviaqa,hotpotqa,musique,"
            "meeting_summarization"
        ),
    )
    parser.add_argument("--minimum-examples", type=int, default=3)
    args = parser.parse_args()

    validate_versions()
    validate_runner()
    datasets = {
        name.strip(): validate_jsonl(
            name.strip(),
            min(args.minimum_examples, 5)
            if name.strip() == "longbench"
            else args.minimum_examples,
        )
        for name in args.benchmarks.split(",")
        if name.strip()
    }
    compressors = validate_compressors()
    print(json.dumps({"status": "ok", "datasets": datasets, **compressors}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
