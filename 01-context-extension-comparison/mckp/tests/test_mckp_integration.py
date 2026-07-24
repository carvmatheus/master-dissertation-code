"""Integração real, executada somente no ambiente experimental local."""
import os

import pytest


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_MCKP_INTEGRATION") != "1",
        reason="requer Ollama, MiniLM e LLMLingua-2 locais",
    ),
]


def test_real_mckp_runner_pipeline(tmp_path, monkeypatch):
    from benchmarks.base import StrategyOutput
    from run_benchmarks import create_mckp_strategy

    monkeypatch.setenv("MCKP_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    strategy = create_mckp_strategy("ollama/llama3.1:8b-instruct-q8_0")
    output = strategy(
        "Paris é a capital da França. Brasília é a capital do Brasil. " * 40,
        "Qual é a capital da França?",
    )

    assert isinstance(output, StrategyOutput)
    assert output.answer
    assert output.compressed_context
    assert output.details["ollama_prompt_eval_count"] > 0
    assert output.details["mckp_audit_record"]["partitions"]
    assert (tmp_path / "audit.jsonl").exists()
