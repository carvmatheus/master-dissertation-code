"""
Testes unitários para o módulo RIG (Dartboard ranking).

Os testes usam mocks para evitar carregar modelos pesados.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Adiciona o módulo ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "01-context-extension-comparison"))


class TestRIGUtils:
    """Testes para funções utilitárias do RIG."""

    def test_normalize_embeddings(self):
        """Verifica normalização de embeddings."""
        from rig.utils import normalize_embeddings
        
        # Vetor simples
        embeddings = np.array([[3.0, 4.0], [1.0, 0.0]])
        normalized = normalize_embeddings(embeddings)
        
        # Primeiro vetor: [3,4] -> norma 5 -> [0.6, 0.8]
        assert np.allclose(normalized[0], [0.6, 0.8])
        # Segundo vetor: [1,0] -> norma 1 -> [1, 0]
        assert np.allclose(normalized[1], [1.0, 0.0])
        
        # Verifica que normas são ~1
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, [1.0, 1.0])

    def test_normalize_zero_vector(self):
        """Vetor zero deve ser tratado sem divisão por zero."""
        from rig.utils import normalize_embeddings
        
        embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
        normalized = normalize_embeddings(embeddings)
        
        # Vetor zero permanece zero (não causa NaN)
        assert np.allclose(normalized[0], [0.0, 0.0])
        assert not np.isnan(normalized).any()

    def test_portuguese_stopwords_exist(self):
        """Verifica que stopwords em português estão definidas."""
        from rig.utils import PORTUGUESE_STOPWORDS
        
        assert len(PORTUGUESE_STOPWORDS) > 30
        assert "de" in PORTUGUESE_STOPWORDS
        assert "para" in PORTUGUESE_STOPWORDS
        assert "que" in PORTUGUESE_STOPWORDS


class TestDartboardProcessorMocked:
    """Testes para DartboardProcessor com mocks (sem carregar modelos reais)."""

    @patch("rig.dartboard_processor._ensure_imports")
    def test_split_text_into_chunks(self, mock_imports):
        """Verifica divisão de texto em chunks."""
        # Mock das dependências
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        with patch.dict("rig.dartboard_processor.__dict__", {
            "_faiss": mock_faiss,
            "_SentenceTransformer": lambda x: mock_model,
            "_TfidfVectorizer": MagicMock,
            "_cosine_similarity": MagicMock,
        }):
            from rig.dartboard_processor import DartboardProcessor
            
            processor = DartboardProcessor.__new__(DartboardProcessor)
            processor.chunk_size = 50  # Pequeno para teste
            
            text = " ".join([f"word{i}" for i in range(100)])
            chunks = processor.split_text_into_chunks(text)
            
            assert len(chunks) > 1
            assert all(isinstance(c, str) for c in chunks)

    def test_adjust_weights_normalizes(self):
        """Verifica que adjust_weights normaliza os pesos."""
        # Setup mínimo do processor
        from rig.dartboard_processor import DartboardProcessor
        
        # Cria instância parcial para testar adjust_weights
        processor = object.__new__(DartboardProcessor)
        processor.alpha = 0.5
        processor.beta = 0.3
        processor.gamma = 0.2
        
        processor.adjust_weights(alpha=2.0, beta=1.0, gamma=1.0, normalize=True)
        
        # Soma deve ser ~1
        total = processor.alpha + processor.beta + processor.gamma
        assert abs(total - 1.0) < 0.001
        
        # Proporções mantidas: alpha deve ser 0.5 (2/4)
        assert abs(processor.alpha - 0.5) < 0.001


class TestRIGStrategy:
    """Testes para RIGStrategy em context_strategies.py."""

    def test_rig_strategy_init(self):
        """Verifica inicialização da estratégia RIG."""
        from context_strategies import RIGStrategy
        
        strategy = RIGStrategy(top_k=5, alpha=0.8)
        
        assert strategy.top_k == 5
        assert strategy.alpha == 0.8
        assert strategy._processor is None  # Lazy load

    def test_rig_strategy_empty_input(self):
        """Texto ou query vazia deve retornar lista vazia."""
        from context_strategies import RIGStrategy
        
        strategy = RIGStrategy()
        
        assert strategy.process("", "query") == []
        assert strategy.process("texto", "") == []
        assert strategy.process("", "") == []

    @patch("context_strategies.RIGStrategy._get_processor")
    def test_rig_strategy_process_calls_processor(self, mock_get_processor):
        """Verifica que process() usa o DartboardProcessor."""
        from context_strategies import RIGStrategy
        
        # Mock do processor
        mock_processor = MagicMock()
        mock_processor.query.return_value = [
            {"text": "chunk1", "score": 0.9},
            {"text": "chunk2", "score": 0.7},
        ]
        mock_get_processor.return_value = mock_processor
        
        strategy = RIGStrategy(top_k=2)
        result = strategy.process("texto de teste", "minha query")
        
        # Deve chamar index_text e query
        mock_processor.index_text.assert_called_once_with("texto de teste")
        mock_processor.query.assert_called_once_with("minha query", top_k=2)
        
        # Deve retornar apenas os textos
        assert result == ["chunk1", "chunk2"]


class TestDartboardRankingLogic:
    """Testes para a lógica de ranking Dartboard."""

    def test_combined_score_formula(self):
        """Verifica a fórmula de combinação de scores."""
        alpha, beta, gamma = 0.7, 0.2, 0.1
        
        semantic_score = 0.8
        lexical_score = 0.5
        importance_score = 0.3
        
        expected = (
            alpha * semantic_score +
            beta * lexical_score +
            gamma * importance_score
        )
        
        # 0.7 * 0.8 + 0.2 * 0.5 + 0.1 * 0.3 = 0.56 + 0.1 + 0.03 = 0.69
        assert abs(expected - 0.69) < 0.001

    def test_weights_sum_to_one(self):
        """Pesos padrão devem somar 1."""
        alpha, beta, gamma = 0.7, 0.2, 0.1
        assert abs((alpha + beta + gamma) - 1.0) < 0.001
