"""
Testes unitários para context_strategies.py
"""
import sys
from pathlib import Path

# Adiciona o módulo ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "01-context-extension-comparison"))

from context_strategies import (
    SlidingWindowStrategy,
    ParallelWindowStrategy,
    RIGStrategy,
    MemGPTStrategy,
)


class TestSlidingWindowStrategy:
    """Testes para a estratégia de janela deslizante."""

    def test_basic_chunking(self):
        """Verifica se o texto é dividido em chunks."""
        strategy = SlidingWindowStrategy(chunk_size=10, overlap=2)
        text = " ".join([f"word{i}" for i in range(25)])  # 25 palavras
        
        chunks = strategy.process(text, "")
        
        assert len(chunks) > 1, "Deveria gerar múltiplos chunks"
        assert all(isinstance(c, str) for c in chunks), "Todos chunks devem ser strings"

    def test_overlap_works(self):
        """Verifica se há sobreposição entre chunks."""
        strategy = SlidingWindowStrategy(chunk_size=10, overlap=3)
        text = " ".join([f"word{i}" for i in range(20)])
        
        chunks = strategy.process(text, "")
        
        # Com overlap, palavras devem aparecer em múltiplos chunks
        if len(chunks) >= 2:
            words_chunk0 = set(chunks[0].split())
            words_chunk1 = set(chunks[1].split())
            overlap = words_chunk0 & words_chunk1
            assert len(overlap) > 0, "Deveria haver palavras em comum (overlap)"

    def test_small_text_single_chunk(self):
        """Texto menor que chunk_size deve retornar um único chunk."""
        strategy = SlidingWindowStrategy(chunk_size=100, overlap=10)
        text = "apenas cinco palavras aqui"
        
        chunks = strategy.process(text, "")
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        """Texto vazio deve retornar lista vazia ou chunk vazio."""
        strategy = SlidingWindowStrategy(chunk_size=10, overlap=2)
        
        chunks = strategy.process("", "")
        
        # Pode retornar [] ou [""] dependendo da implementação
        assert len(chunks) <= 1


class TestParallelWindowStrategy:
    """Testes para a estratégia de janela paralela (MapReduce)."""

    def test_strict_division(self):
        """Verifica divisão estrita sem overlap."""
        strategy = ParallelWindowStrategy(chunk_size=5)
        text = " ".join([f"w{i}" for i in range(15)])  # 15 palavras
        
        chunks = strategy.process(text, "")
        
        assert len(chunks) == 3, "15 palavras / 5 = 3 chunks"

    def test_uneven_division(self):
        """Verifica comportamento com divisão não exata."""
        strategy = ParallelWindowStrategy(chunk_size=4)
        text = " ".join([f"w{i}" for i in range(10)])  # 10 palavras
        
        chunks = strategy.process(text, "")
        
        # 10 / 4 = 2.5, então 3 chunks (último menor)
        assert len(chunks) == 3
        assert len(chunks[-1].split()) == 2  # Últimas 2 palavras

    def test_no_overlap(self):
        """Verifica que não há overlap entre chunks."""
        strategy = ParallelWindowStrategy(chunk_size=5)
        text = " ".join([f"unique{i}" for i in range(15)])
        
        chunks = strategy.process(text, "")
        
        all_words = []
        for chunk in chunks:
            all_words.extend(chunk.split())
        
        # Não deve haver duplicatas
        assert len(all_words) == len(set(all_words)), "Não deveria haver overlap"


class TestPlaceholderStrategies:
    """Testes para estratégias ainda não implementadas."""

    def test_memgpt_returns_todo(self):
        """MemGPTStrategy deve retornar placeholder TODO."""
        strategy = MemGPTStrategy()
        result = strategy.process("qualquer texto", "query")
        
        assert len(result) == 1
        assert "TODO" in result[0]


class TestRIGStrategyBasic:
    """Testes básicos para RIGStrategy (sem carregar modelos)."""

    def test_rig_strategy_initialization(self):
        """Verifica que RIGStrategy pode ser inicializada."""
        strategy = RIGStrategy(top_k=5, alpha=0.8, beta=0.15, gamma=0.05)
        
        assert strategy.top_k == 5
        assert strategy.alpha == 0.8
        assert strategy.beta == 0.15
        assert strategy.gamma == 0.05

    def test_rig_empty_input_returns_empty(self):
        """Texto ou query vazia deve retornar lista vazia."""
        strategy = RIGStrategy()
        
        assert strategy.process("", "query") == []
        assert strategy.process("texto", "") == []
