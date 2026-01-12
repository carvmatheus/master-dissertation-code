"""
Utilidades para o módulo RIG.
"""
import numpy as np
from typing import List

# Stopwords em português para o TF-IDF
PORTUGUESE_STOPWORDS: List[str] = [
    # Artigos
    "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "as", "às",
    "da", "das", "de", "dela", "delas", "dele", "deles", "do", "dos", "duas",
    "esta", "estas", "este", "estes", "está", "estás", "o", "os", "um", "uma",
    "umas", "uns",
    # Preposições/contrações
    "com", "como", "contra", "desde", "em", "entre", "para", "perante", "por",
    "sem", "sob", "sobre", "trás", "pela", "pelas", "pelo", "pelos", "num",
    "numa", "nuns", "numas", "dum", "duma", "duns", "dumas",
    # Pronomes
    "ele", "eles", "eu", "lhe", "lhes", "me", "meu", "meus", "minha", "minhas",
    "nós", "se", "seu", "seus", "sua", "suas", "te", "tu", "tua", "tuas",
    "você", "vocês", "vos",
    # Conjunções
    "e", "mas", "nem", "ou", "porém", "que", "quer", "então", "todavia",
    # Advérbios/interjeições
    "agora", "aí", "ainda", "ali", "amanhã", "antes", "aqui", "assim", "bem",
    "cedo", "depois", "hoje", "logo", "mais", "mal", "melhor", "menos", "muito",
    "não", "onde", "ontem", "pra", "qual", "quando", "quanto", "quê", "sim",
    "talvez", "tão", "tarde", "tem", "têm", "já", "só",
    # Outros
    "etc", "exemplo", "isso", "isto", "outro", "outros", "qualquer", "seja",
    "também", "outra", "ser", "há", "outras", "tempo", "vez", "vezes", "via",
]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normaliza embeddings para norma unitária (L2).
    
    Args:
        embeddings: Array de embeddings shape (n_samples, dim)
        
    Returns:
        Embeddings normalizados
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Evita divisão por zero
    return embeddings / norms
