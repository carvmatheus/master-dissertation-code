"""Contratos entre cache, serialização, orçamento e estratégia MCKP."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mckp.options as options_module
from mckp.models import CompressionOption, MCKPConfig, Partition
from mckp.options import OptionGenerator
from mckp.reconstructor import reconstruct, serialized_option_cost
from mckp.strategy import MCKPStrategy
from mckp.tokenization import TiktokenTokenCounter


class QueryCompressor:
    query_dependent = True

    def compress(self, text, param, query=None):
        return f"resultado para {query}"


def test_query_dependent_cache_includes_query(monkeypatch):
    monkeypatch.setattr(options_module, "fidelity", lambda *args, **kwargs: 0.5)
    generator = OptionGenerator(MCKPConfig())
    generator._compressors["sentence_extractive"] = QueryCompressor()

    first = generator._materialize(
        "contexto compartilhado", "sentence_extractive", 0.5, "pergunta um"
    )
    second = generator._materialize(
        "contexto compartilhado", "sentence_extractive", 0.5, "pergunta dois"
    )

    assert first[0] == "resultado para pergunta um"
    assert second[0] == "resultado para pergunta dois"


def test_reconstruction_never_exceeds_serialized_option_cost():
    counter = TiktokenTokenCounter()
    texts = ["alpha", "", "beta"]
    chosen = [
        CompressionOption(
            partition_index=index,
            compressor="omission" if not text else "identity",
            param=0.0 if not text else 1.0,
            text=text,
            token_cost=serialized_option_cost(text, index, counter),
            fidelity=0.0 if not text else 1.0,
            importance=1.0,
        )
        for index, text in enumerate(texts)
    ]

    rebuilt = reconstruct(chosen)
    assert "omitidos" not in rebuilt
    assert counter.count(rebuilt) <= sum(option.token_cost for option in chosen)


def test_compressor_failure_is_observable():
    config = MCKPConfig(
        option_set=[
            {"compressor": "compressor_inexistente", "param": 0.5},
            {"compressor": "identity", "param": 1.0},
            {"compressor": "omission", "param": 0.0},
        ]
    )
    generator = OptionGenerator(config)
    partition = options_module.Partition(index=0, text="texto", importance=1.0)

    generated = generator.generate([partition], query="consulta")

    assert {option.compressor for option in generated[0]} == {"identity", "omission"}
    assert len(generator.failures) == 1
    assert generator.failures[0]["compressor"] == "compressor_inexistente"


def test_required_compressor_failure_aborts_run():
    config = MCKPConfig(
        option_set=[{"compressor": "compressor_inexistente", "param": 0.5}],
        required_compressors=["compressor_inexistente"],
    )
    generator = OptionGenerator(config)

    with pytest.raises(RuntimeError, match="compressores obrigatórios indisponíveis"):
        generator.generate([Partition(index=0, text="texto", importance=1.0)], "consulta")


def test_required_partition_cannot_be_omitted():
    config = MCKPConfig(
        budget_tokens=100,
        option_set=[
            {"compressor": "identity", "param": 1.0},
            {"compressor": "omission", "param": 0.0},
        ],
    )
    strategy = MCKPStrategy(config)
    compressed = strategy.process(
        "conteúdo obrigatório", "", required_partition_indices=[0]
    )[0]

    assert "conteúdo obrigatório" in compressed
    assert strategy.last_diagnostics["chosen_compressors"] == ["identity"]


def test_audit_log_records_partition_decisions(tmp_path):
    audit_path = tmp_path / "mckp.jsonl"
    config = MCKPConfig(
        budget_tokens=100,
        audit_log_path=str(audit_path),
        option_set=[
            {"compressor": "identity", "param": 1.0},
            {"compressor": "omission", "param": 0.0},
        ],
    )
    strategy = MCKPStrategy(config)
    strategy.process("fato 42 importante", "")

    import json

    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert record["partitions"][0]["importance_components"]
    assert record["partitions"][0]["options"]
    assert record["partitions"][0]["chosen"]["compressor"]
    assert "unused_tokens" in record["budget"]


@pytest.mark.parametrize("model_context", [8192, 32768])
def test_context_budget_uses_each_experimental_window(model_context):
    config = MCKPConfig(
        model_context_tokens=model_context,
        output_tokens=500,
        token_safety_margin=256,
        option_set=[{"compressor": "identity", "param": 1.0}],
    )
    strategy = MCKPStrategy(config)
    budget = strategy._budget("Qual é a resposta?")
    prompt_overhead = strategy.token_counter.count(
        strategy.prompt_builder("", "Qual é a resposta?")
    )

    assert budget == model_context - 500 - 256 - prompt_overhead


def test_end_to_end_context_respects_hard_budget_without_markers():
    config = MCKPConfig(
        max_partition_tokens=12,
        budget_tokens=40,
        budget_bucket=1,
        option_set=[
            {"compressor": "identity", "param": 1.0},
            {"compressor": "omission", "param": 0.0},
        ],
    )
    strategy = MCKPStrategy(config)
    text = "\n\n".join(
        f"Partição {index} contém informação relevante e alguns detalhes adicionais."
        for index in range(20)
    )

    compressed = strategy.process(text, query="")[0]

    assert strategy.last_diagnostics["actual_context_tokens"] <= 40
    assert strategy.token_counter.count(compressed) <= 40
    assert "trechos omitidos" not in compressed
