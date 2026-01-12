from typing import List, Generator

class ContextStrategy:
    def process(self, text: str, query: str) -> List[str]:
        """
        Processa o texto e retorna uma lista de contextos a serem enviados ao LLM.
        Algumas estratégias retornam apenas 1 bloco (ex: compressão), 
        outras retornam múltiplos (ex: janela deslizante).
        """
        raise NotImplementedError

class SlidingWindowStrategy(ContextStrategy):
    """
    Segmentação de Contexto e Janela Deslizante.
    Divide o texto em chunks com sobreposição (overlap).
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process(self, text: str, query: str) -> List[str]:
        # Implementação simples baseada em caracteres/palavras para demonstração
        # Idealmente usaria contagem de tokens
        words = text.split()
        chunks = []
        
        step = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_str = " ".join(chunk_words)
            chunks.append(chunk_str)
            if i + self.chunk_size >= len(words):
                break
        
        return chunks

class ParallelWindowStrategy(ContextStrategy):
    """
    Similar à Janela Deslizante, mas foca em preparar os chunks 
    para serem enviados em chamadas paralelas (MapReduce).
    """
    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size

    def process(self, text: str, query: str) -> List[str]:
        # Divide estritamente sem muito overlap para processamento paralelo
        words = text.split()
        return [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

# RIG - Reinforced Information Gain com Dartboard ranking
class RIGStrategy(ContextStrategy):
    """
    Estratégia baseada em Dartboard ranking.
    
    Combina três sinais para recuperar os chunks mais relevantes:
    - Similaridade semântica (embeddings)
    - Similaridade lexical (TF-IDF)
    - Importância do documento
    
    Requer: faiss-cpu, sentence-transformers, scikit-learn
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
    ):
        """
        Args:
            embedding_model: Modelo SentenceTransformer para embeddings
            top_k: Número de chunks a retornar
            alpha: Peso para similaridade semântica
            beta: Peso para similaridade lexical
            gamma: Peso para importância
        """
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._processor = None
    
    def _get_processor(self):
        """Lazy load do processador Dartboard."""
        if self._processor is None:
            try:
                from rig import DartboardProcessor
            except ImportError:
                from .rig import DartboardProcessor
            
            self._processor = DartboardProcessor(
                embedding_model=self.embedding_model,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )
        return self._processor
    
    def process(self, text: str, query: str) -> List[str]:
        """
        Indexa o texto e retorna os chunks mais relevantes para a query.
        
        Args:
            text: Texto completo a ser indexado
            query: Pergunta do usuário
            
        Returns:
            Lista dos top_k chunks mais relevantes
        """
        if not text or not query:
            return []
        
        processor = self._get_processor()
        
        # Indexa o texto (divide em chunks e gera embeddings)
        processor.index_text(text)
        
        # Busca chunks relevantes
        results = processor.query(query, top_k=self.top_k)
        
        return [r["text"] for r in results]
    
    def get_detailed_results(self, text: str, query: str) -> List[dict]:
        """
        Versão detalhada que retorna scores junto com os chunks.
        
        Returns:
            Lista de dicts com text, score, semantic_score, lexical_score, importance_score
        """
        if not text or not query:
            return []
        
        processor = self._get_processor()
        processor.index_text(text)
        return processor.query(query, top_k=self.top_k)

class MemGPTStrategy(ContextStrategy):
    def process(self, text: str, query: str) -> List[str]:
        return ["(TODO: Implementar lógica de Memória Hierárquica)"]
