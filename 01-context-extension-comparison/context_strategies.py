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

# Placeholder para futuras implementações
class RIGStrategy(ContextStrategy):
    def process(self, text: str, query: str) -> List[str]:
        return ["(TODO: Implementar lógica de Relevant Information Gain)"]

class MemGPTStrategy(ContextStrategy):
    def process(self, text: str, query: str) -> List[str]:
        return ["(TODO: Implementar lógica de Memória Hierárquica)"]
