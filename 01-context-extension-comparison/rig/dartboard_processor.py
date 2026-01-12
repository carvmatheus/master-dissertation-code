"""
Dartboard Processor - RAG com ranking híbrido.

Combina três sinais para ranquear documentos:
1. Similaridade semântica (embeddings via SentenceTransformer)
2. Similaridade lexical (TF-IDF)
3. Importância/popularidade do documento
"""
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from .utils import PORTUGUESE_STOPWORDS, normalize_embeddings

# Lazy imports para dependências pesadas
_faiss = None
_SentenceTransformer = None
_TfidfVectorizer = None
_cosine_similarity = None


def _ensure_imports():
    """Lazy import das dependências pesadas."""
    global _faiss, _SentenceTransformer, _TfidfVectorizer, _cosine_similarity
    
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu não instalado. Rode: pip install faiss-cpu")
    
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SentenceTransformer = SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers não instalado. Rode: pip install sentence-transformers")
    
    if _TfidfVectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        _TfidfVectorizer = TfidfVectorizer
        _cosine_similarity = cosine_similarity


class DartboardProcessor:
    """
    Processador de documentos com ranking Dartboard.
    
    Combina embeddings semânticos, TF-IDF lexical e scores de importância
    para ranquear chunks de documentos de forma mais precisa que embeddings puros.
    
    Attributes:
        alpha: Peso para similaridade semântica (default 0.7)
        beta: Peso para similaridade lexical (default 0.2)
        gamma: Peso para importância (default 0.1)
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 2000,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
    ):
        """
        Inicializa o processador Dartboard.
        
        Args:
            embedding_model: Nome do modelo SentenceTransformer
            chunk_size: Tamanho máximo de cada chunk em caracteres
            alpha: Peso para embeddings semânticos
            beta: Peso para similaridade lexical (TF-IDF)
            gamma: Peso para importância/popularidade
        """
        _ensure_imports()
        
        print(f"Carregando modelo de embeddings: {embedding_model}...")
        self.embedding_model = _SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Índice FAISS para busca por similaridade
        self.index = _faiss.IndexFlatIP(self.dimension)
        
        # Armazenamento
        self.document_chunks: List[str] = []
        self.document_embeddings: Optional[np.ndarray] = None
        
        # TF-IDF
        self.tfidf_vectorizer: Optional[Any] = None
        self.tfidf_matrix: Optional[Any] = None
        
        # Scores de importância
        self.importance_scores: Optional[np.ndarray] = None
        
        # Configuração
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Divide texto em chunks de tamanho configurado.
        
        Args:
            text: Texto completo
            
        Returns:
            Lista de chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1
            if current_size + word_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Gera embeddings para lista de textos."""
        print(f"Gerando embeddings para {len(texts)} chunk(s)...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=show_progress)
        return embeddings
    
    def _build_lexical_index(self, texts: List[str]) -> None:
        """Constrói índice TF-IDF para busca lexical."""
        print("Construindo índice lexical (TF-IDF)...")
        self.tfidf_vectorizer = _TfidfVectorizer(
            lowercase=True,
            stop_words=PORTUGUESE_STOPWORDS
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def _calculate_importance_scores(self, texts: List[str]) -> np.ndarray:
        """
        Calcula scores de importância baseado no tamanho dos chunks.
        
        Pode ser estendido para usar métricas de popularidade/acesso.
        """
        importance = np.array([len(text.split()) for text in texts], dtype=np.float32)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        return importance
    
    def index_chunks(self, chunks: List[str]) -> None:
        """
        Indexa uma lista de chunks já divididos.
        
        Args:
            chunks: Lista de strings (chunks de texto)
        """
        if not chunks:
            print("Lista de chunks vazia.")
            return
        
        self.document_chunks = chunks
        
        # Gera e normaliza embeddings
        embeddings = self._generate_embeddings(chunks)
        normalized_embeddings = normalize_embeddings(embeddings)
        self.document_embeddings = normalized_embeddings.copy()
        
        # Índices auxiliares
        self._build_lexical_index(chunks)
        self.importance_scores = self._calculate_importance_scores(chunks)
        
        # Adiciona ao FAISS
        self.index.reset()
        self.index.add(normalized_embeddings.astype(np.float32))
        
        print(f"Indexados {len(chunks)} chunks com sucesso!")
    
    def index_text(self, text: str) -> None:
        """
        Divide texto em chunks e indexa.
        
        Args:
            text: Texto completo para indexar
        """
        chunks = self.split_text_into_chunks(text)
        self.index_chunks(chunks)
    
    def dartboard_ranking(
        self,
        query: str,
        top_k: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Executa ranking Dartboard combinando os três sinais.
        
        Args:
            query: Pergunta do usuário
            top_k: Número de resultados iniciais para reranquear
            
        Returns:
            Lista de dicts com chunk_id, score, text e scores individuais
        """
        if self.index.ntotal == 0:
            print("Nenhum documento indexado.")
            return []
        
        # 1. Busca semântica inicial
        query_embedding = self.embedding_model.encode([query])
        normalized_query = normalize_embeddings(query_embedding)
        distances, indices = self.index.search(
            normalized_query.astype(np.float32), top_k
        )
        
        # 2. Score lexical (TF-IDF)
        lexical_scores = np.zeros(len(indices[0]))
        if self.tfidf_vectorizer is not None:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            for i, idx in enumerate(indices[0]):
                lexical_scores[i] = _cosine_similarity(
                    query_tfidf, self.tfidf_matrix[idx].reshape(1, -1)
                ).item()
        
        # 3. Score de importância
        importance_scores = np.zeros(len(indices[0]))
        if self.importance_scores is not None:
            for i, idx in enumerate(indices[0]):
                importance_scores[i] = self.importance_scores[idx]
        
        # 4. Combinação Dartboard
        combined_scores = (
            self.alpha * distances[0] +
            self.beta * lexical_scores +
            self.gamma * importance_scores
        )
        
        # 5. Reordena por score combinado
        reranked_indices = np.argsort(-combined_scores)
        
        results = []
        for i in reranked_indices:
            doc_idx = indices[0][i]
            results.append({
                "chunk_id": int(doc_idx),
                "score": float(combined_scores[i]),
                "text": self.document_chunks[doc_idx],
                "semantic_score": float(distances[0][i]),
                "lexical_score": float(lexical_scores[i]),
                "importance_score": float(importance_scores[i]),
            })
        
        return results
    
    def filter_diverse_results(
        self,
        results: List[Dict[str, Any]],
        diversity_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Filtra resultados para remover chunks muito similares.
        
        Args:
            results: Lista de resultados do ranking
            diversity_threshold: Limiar de similaridade para considerar redundante
            
        Returns:
            Lista filtrada
        """
        if self.document_embeddings is None:
            return results
        
        selected = []
        selected_ids = []
        
        for result in results:
            chunk_id = result["chunk_id"]
            candidate_embedding = self.document_embeddings[chunk_id]
            
            is_redundant = False
            for sid in selected_ids:
                sim = np.dot(candidate_embedding, self.document_embeddings[sid])
                if sim > diversity_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(result)
                selected_ids.append(chunk_id)
        
        return selected
    
    def query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Busca chunks relevantes para uma query.
        
        Args:
            query: Pergunta do usuário
            top_k: Número de resultados finais
            
        Returns:
            Lista dos top_k chunks mais relevantes
        """
        retrieval_k = max(top_k * 3, 30)
        initial_results = self.dartboard_ranking(query, retrieval_k)
        diverse_results = self.filter_diverse_results(initial_results)
        return diverse_results[:top_k]
    
    def get_context_for_llm(self, query: str, top_k: int = 3) -> str:
        """
        Retorna contexto formatado para enviar a um LLM.
        
        Args:
            query: Pergunta do usuário
            top_k: Número de chunks a incluir
            
        Returns:
            String com contexto formatado
        """
        results = self.query(query, top_k)
        if not results:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(results):
            context_parts.append(
                f"[Documento {i+1} - Score: {chunk['score']:.4f}]\n{chunk['text']}"
            )
        
        return "\n\n".join(context_parts)
    
    def adjust_weights(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        normalize: bool = True
    ) -> None:
        """
        Ajusta os pesos do ranking Dartboard.
        
        Args:
            alpha: Peso para embeddings semânticos
            beta: Peso para similaridade lexical
            gamma: Peso para importância
            normalize: Se True, normaliza para soma = 1
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        
        if normalize:
            total = self.alpha + self.beta + self.gamma
            if total > 0:
                self.alpha /= total
                self.beta /= total
                self.gamma /= total
        
        print(f"Pesos ajustados: alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}")
    
    def save_state(
        self,
        output_dir: str = ".",
        prefix: str = "dartboard"
    ) -> None:
        """
        Salva estado do processador para disco.
        
        Args:
            output_dir: Diretório de saída
            prefix: Prefixo para os arquivos
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chunks
        with open(output_dir / f"{prefix}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.document_chunks, f, ensure_ascii=False)
        
        # FAISS index
        _faiss.write_index(self.index, str(output_dir / f"{prefix}_index.bin"))
        
        # TF-IDF
        if self.tfidf_vectorizer is not None:
            with open(output_dir / f"{prefix}_tfidf.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        # Importance scores
        if self.importance_scores is not None:
            np.save(output_dir / f"{prefix}_importance.npy", self.importance_scores)
        
        print(f"Estado salvo em {output_dir}/")
    
    def load_state(
        self,
        input_dir: str = ".",
        prefix: str = "dartboard"
    ) -> bool:
        """
        Carrega estado do processador do disco.
        
        Args:
            input_dir: Diretório de entrada
            prefix: Prefixo dos arquivos
            
        Returns:
            True se carregou com sucesso
        """
        input_dir = Path(input_dir)
        chunks_file = input_dir / f"{prefix}_chunks.json"
        index_file = input_dir / f"{prefix}_index.bin"
        
        if not chunks_file.exists() or not index_file.exists():
            return False
        
        # Chunks
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.document_chunks = json.load(f)
        
        # FAISS index
        self.index = _faiss.read_index(str(index_file))
        
        # Regenera embeddings
        if self.document_chunks:
            embeddings = self._generate_embeddings(self.document_chunks, show_progress=True)
            self.document_embeddings = normalize_embeddings(embeddings)
            
            # TF-IDF
            tfidf_file = input_dir / f"{prefix}_tfidf.pkl"
            if tfidf_file.exists():
                with open(tfidf_file, "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)
                self.tfidf_matrix = self.tfidf_vectorizer.transform(self.document_chunks)
            else:
                self._build_lexical_index(self.document_chunks)
            
            # Importance
            importance_file = input_dir / f"{prefix}_importance.npy"
            if importance_file.exists():
                self.importance_scores = np.load(importance_file)
            else:
                self.importance_scores = self._calculate_importance_scores(self.document_chunks)
        
        print(f"Estado carregado: {len(self.document_chunks)} chunks disponíveis.")
        return True
    
    def clear(self) -> None:
        """Limpa todos os dados indexados."""
        self.document_chunks = []
        self.document_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.importance_scores = None
        self.index = _faiss.IndexFlatIP(self.dimension)
        print("Memória limpa.")
