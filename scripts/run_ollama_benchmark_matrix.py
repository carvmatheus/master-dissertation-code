#!/usr/bin/env python3
"""Executa modelos Ollama sequencialmente para uma ou mais janelas de contexto."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RUNNER = REPO_ROOT / "01-context-extension-comparison" / "run_benchmarks.py"
MCKP_PREFLIGHT = REPO_ROOT / "scripts" / "preflight_mckp.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tests" / "ollama-local" / "ollama_benchmark_runs"
DEFAULT_MODEL = "llama3.1:8b-instruct-q8_0"
DEFAULT_BENCHMARKS = (
    "longbench,zeroscrolls,naturalquestions,triviaqa,hotpotqa,musique,"
    "meeting_summarization"
)


def api_request(path: str, payload: dict | None = None, timeout: int = 30) -> dict:
    url = f"http://localhost:11434{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def server_is_ready() -> bool:
    try:
        api_request("/api/version", timeout=2)
        return True
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def ensure_server(output_root: Path) -> subprocess.Popen | None:
    if server_is_ready():
        return None

    output_root.mkdir(parents=True, exist_ok=True)
    log_file = (output_root / "ollama-server.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(30):
        if server_is_ready():
            return process
        if process.poll() is not None:
            raise RuntimeError(f"ollama serve encerrou com código {process.returncode}")
        time.sleep(1)
    raise TimeoutError("O servidor Ollama não respondeu em 30 segundos")


def model_slug(model: str) -> str:
    return model.replace("hf.co/", "").replace("/", "_").replace(":", "-")


def declared_context_length(model_info: dict) -> int | None:
    values = [
        value
        for key, value in model_info.items()
        if key.endswith(".context_length") and isinstance(value, int)
    ]
    return max(values) if values else None


def save_model_metadata(model: str, output_root: Path) -> int | None:
    response = api_request("/api/show", {"model": model})
    info = response.get("model_info", {})
    context_length = declared_context_length(info)
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "details": response.get("details", {}),
        "capabilities": response.get("capabilities", []),
        "parameters": response.get("parameters", ""),
        "declared_context_length": context_length,
        "context_length_fields": {
            key: value for key, value in info.items() if key.endswith(".context_length")
        },
    }
    metadata_dir = output_root / "models"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / f"{model_slug(model)}.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return context_length


def unload_model(model: str) -> None:
    # Mata o llama-server do modelo e libera toda a RAM; o `ollama serve`
    # (leve, gerenciado pelo Ollama.app) permanece de pé.
    subprocess.run(["ollama", "stop", model], check=False, timeout=60)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    output_dir: Path, command: list[str], model: str, num_ctx: int, args: argparse.Namespace
) -> None:
    data_dir = REPO_ROOT / "data" / "benchmarks"
    benchmarks = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    dataset_hashes = {
        name: file_sha256(data_dir / f"{name}.jsonl")
        for name in benchmarks
        if (data_dir / f"{name}.jsonl").exists()
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "num_ctx": num_ctx,
        "strategies": args.strategies.split(","),
        "benchmarks": benchmarks,
        "quick": not args.full,
        "examples_per_benchmark": args.examples_per_benchmark,
        "mckp_mu": args.mckp_mu,
        "mckp_distance": args.mckp_distance,
        "mckp_budget_bucket": args.mckp_budget_bucket,
        "command": command,
        "dataset_sha256": dataset_hashes,
        "mckp_config_sha256": file_sha256(
            REPO_ROOT / "01-context-extension-comparison" / "mckp" / "config.json"
        ),
        "requirements_sha256": file_sha256(REPO_ROOT / "requirements.txt"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def run_and_tee(command: list[str], cwd: Path, log_path: Path) -> int:
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def save_context_comparison(output_root: Path) -> Path:
    rows = []
    benchmark_names = set()
    for result_path in sorted((output_root / "runs").glob("*/ctx-*/benchmark_results.json")):
        with result_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        valid_results = [result for result in report.get("results", []) if not result.get("ollama_error")]
        if not valid_results:
            continue
        benchmark_names.update(result["benchmark"] for result in valid_results)
        for strategy in sorted({result["strategy"] for result in valid_results}):
            strategy_results = [result for result in valid_results if result["strategy"] == strategy]
            benchmark_scores = {}
            for benchmark in {result["benchmark"] for result in strategy_results}:
                scores = [
                    float(result["score"])
                    for result in strategy_results
                    if result["benchmark"] == benchmark
                ]
                benchmark_scores[benchmark] = sum(scores) / len(scores)
            rows.append(
                {
                    "model": result_path.parents[1].name,
                    "num_ctx": int(result_path.parent.name.removeprefix("ctx-")),
                    "strategy": strategy,
                    "avg_score": sum(float(result["score"]) for result in strategy_results)
                    / len(strategy_results),
                    "avg_latency_ms": sum(
                        float(result["latency_ms"]) for result in strategy_results
                    )
                    / len(strategy_results),
                    "num_tests": len(strategy_results),
                    "benchmarks": benchmark_scores,
                }
            )

    output_path = output_root / "context_comparison.csv"
    ordered_benchmarks = sorted(benchmark_names)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "num_ctx", "strategy", "avg_score", "avg_latency_ms", "num_tests"]
            + [f"score_{name}" for name in ordered_benchmarks]
        )
        for row in sorted(
            rows,
            key=lambda item: (item["model"], item["num_ctx"], item["strategy"]),
        ):
            writer.writerow(
                [
                    row["model"],
                    row["num_ctx"],
                    row["strategy"],
                    row["avg_score"],
                    row["avg_latency_ms"],
                    row["num_tests"],
                ]
                + [row["benchmarks"].get(name) for name in ordered_benchmarks]
            )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=DEFAULT_MODEL, help="Modelos separados por vírgula")
    parser.add_argument(
        "--contexts",
        default="8192",
        help="Valores de num_ctx em ordem crescente, separados por vírgula",
    )
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--strategies", default="raw")
    parser.add_argument("--examples-per-benchmark", type=int, default=None)
    parser.add_argument("--mckp-mu", type=float, default=None)
    parser.add_argument(
        "--mckp-distance",
        choices=["compressor_family", "param_diff", "none"],
        default=None,
    )
    parser.add_argument("--mckp-budget-bucket", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--full", action="store_true", help="Executa configuração completa")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Atualiza context_comparison.csv sem executar modelos",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Pula combinações que já possuem benchmark_results.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    models = [model.strip().removeprefix("ollama/") for model in args.models.split(",")]
    contexts = sorted({int(value.strip()) for value in args.contexts.split(",")})
    if not models or not contexts or any(value <= 756 for value in contexts):
        raise ValueError("Informe modelos e contextos maiores que 756 tokens")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(f"Comparação salva em: {save_context_comparison(args.output_root)}")
        return 0

    if any(name.strip().startswith("mckp") for name in args.strategies.split(",")):
        minimum = "20" if args.full else "3"
        subprocess.run(
            [
                sys.executable,
                str(MCKP_PREFLIGHT),
                "--benchmarks",
                args.benchmarks,
                "--minimum-examples",
                minimum,
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    ensure_server(args.output_root)
    failures = 0

    for model in models:
        declared_limit = save_model_metadata(model, args.output_root)
        try:
            for num_ctx in contexts:
                if declared_limit is not None and num_ctx > declared_limit:
                    print(
                        f"SKIP {model}: num_ctx={num_ctx} excede o limite declarado "
                        f"de {declared_limit}"
                    )
                    continue

                output_dir = args.output_root / "runs" / model_slug(model) / f"ctx-{num_ctx}"
                result_path = output_dir / "benchmark_results.json"
                if result_path.exists():
                    if args.skip_completed:
                        print(f"SKIP concluído: {result_path}")
                        continue
                    raise FileExistsError(
                        f"resultado já existe: {result_path}. "
                        "Use outra --output-root ou --skip-completed."
                    )
                command = [
                    sys.executable,
                    str(BENCHMARK_RUNNER),
                    "--models",
                    f"ollama/{model}",
                    "--strategies",
                    args.strategies,
                    "--benchmarks",
                    args.benchmarks,
                    "--ollama-num-ctx",
                    str(num_ctx),
                    "--output-dir",
                    str(output_dir),
                ]
                if not args.full:
                    command.append("--quick")
                if args.examples_per_benchmark is not None:
                    command.extend(
                        ["--examples-per-benchmark", str(args.examples_per_benchmark)]
                    )
                if args.mckp_mu is not None:
                    command.extend(["--mckp-mu", str(args.mckp_mu)])
                if args.mckp_distance is not None:
                    command.extend(["--mckp-distance", args.mckp_distance])
                if args.mckp_budget_bucket is not None:
                    command.extend(
                        ["--mckp-budget-bucket", str(args.mckp_budget_bucket)]
                    )

                print(f"\nRUN {model} com num_ctx={num_ctx}")
                write_run_manifest(output_dir, command, model, num_ctx, args)
                returncode = run_and_tee(command, REPO_ROOT, output_dir / "run.log")
                if returncode != 0:
                    failures += 1
                    print(f"FALHA {model} num_ctx={num_ctx}: código {returncode}")
        finally:
            # Descarrega o modelo assim que ele não é mais usado (libera a RAM).
            unload_model(model)

    print(f"Comparação salva em: {save_context_comparison(args.output_root)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
