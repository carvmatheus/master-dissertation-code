"""
Disjunção e cobertura do particionador estrutural.

As partições devem cobrir todo o texto sem sobreposição, ao contrário da
segmentação por janela com overlap do baseline RSAW. A verificação compara o
multiconjunto de palavras do texto original com o das partições.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mckp.models import MCKPConfig
from mckp.partitioner import StructuralPartitioner, WholeContextPartitioner

TEXT = (
    "O primeiro paragrafo trata da introducao do problema de compressao. "
    "Ele apresenta o contexto geral e a motivacao inicial.\n\n"
    "O segundo paragrafo descreve a metodologia adotada no experimento. "
    "Aqui os procedimentos sao detalhados passo a passo com cuidado.\n\n"
    "O terceiro paragrafo discute os resultados obtidos nas medicoes. "
    "As metricas de custo e qualidade sao comparadas entre estrategias.\n\n"
    "O quarto paragrafo conclui o argumento e aponta trabalhos futuros. "
    "Ele resume as contribuicoes e as limitacoes encontradas."
)


def test_disjoint_and_covering():
    cfg = MCKPConfig(max_partition_tokens=40)
    partitions = StructuralPartitioner(cfg).partition(TEXT)

    assert len(partitions) >= 2
    for i, p in enumerate(partitions):
        assert p.index == i

    original_words = Counter(TEXT.split())
    partition_words: Counter = Counter()
    for p in partitions:
        partition_words.update(p.text.split())

    # Sem duplicacao (disjuncao) e sem perda (cobertura).
    assert partition_words == original_words


def test_respects_token_ceiling():
    cfg = MCKPConfig(max_partition_tokens=40)
    partitions = StructuralPartitioner(cfg).partition(TEXT)
    from mckp.fidelity import count_tokens

    for p in partitions:
        assert count_tokens(p.text) <= cfg.max_partition_tokens


def test_splits_single_oversized_sentence_by_tokens():
    cfg = MCKPConfig(max_partition_tokens=12)
    text = " ".join(f"palavra{i}" for i in range(100)) + "."
    partitions = StructuralPartitioner(cfg).partition(text)
    from mckp.fidelity import count_tokens

    assert len(partitions) > 1
    assert all(count_tokens(part.text) <= cfg.max_partition_tokens for part in partitions)
    assert Counter(" ".join(part.text for part in partitions).split()) == Counter(text.split())


def test_single_partition_for_short_text():
    cfg = MCKPConfig(max_partition_tokens=400)
    partitions = StructuralPartitioner(cfg).partition("Frase curta unica.")
    assert len(partitions) == 1
    assert partitions[0].text == "Frase curta unica."


def test_whole_context_partitioner_returns_single_partition():
    partitions = WholeContextPartitioner().partition(TEXT)
    assert len(partitions) == 1
    assert partitions[0].kind == "whole_context"
    assert partitions[0].text == TEXT
