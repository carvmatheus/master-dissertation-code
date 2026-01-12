"""
RULER Benchmark

Framework que mede o "tamanho real de contexto" de LLMs através de testes
de recuperação de informação inserida em posições variadas.

Expõe quantitativamente:
- Fenômeno "lost in the middle"
- Degradação posicional
- Limite efetivo de contexto (vs. nominal)

Baseado em: Hsieh et al., 2024 - "RULER: What's the Real Context Size of Your LLM?"
"""
import random
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseBenchmark, TestCase


# Templates de fatos para inserir
FACT_TEMPLATES = [
    {
        "template": "O identificador único do registro {id} é {value}.",
        "query_template": "Qual é o identificador único do registro {id}?",
        "value_generator": lambda rng: f"ID-{rng.randint(10000, 99999)}"
    },
    {
        "template": "A chave de acesso para o sistema {id} é {value}.",
        "query_template": "Qual é a chave de acesso para o sistema {id}?",
        "value_generator": lambda rng: f"KEY-{rng.randint(1000, 9999)}-{rng.choice(['ALPHA', 'BETA', 'GAMMA'])}"
    },
    {
        "template": "O valor do parâmetro {id} foi configurado como {value}.",
        "query_template": "Qual é o valor do parâmetro {id}?",
        "value_generator": lambda rng: f"{rng.randint(1, 100)}.{rng.randint(0, 99):02d}"
    },
]

# Texto de preenchimento com estrutura similar
DISTRACTOR_TEMPLATES = [
    "O sistema processou {n} registros no último ciclo de execução.",
    "A taxa de utilização do servidor alcançou {n}% durante o pico.",
    "Foram identificados {n} eventos de segurança na última auditoria.",
    "O tempo médio de resposta foi de {n} milissegundos.",
    "A base de dados contém {n} milhões de registros ativos.",
    "O processo de backup foi concluído em {n} minutos.",
    "A capacidade de armazenamento utilizada é de {n}%.",
    "Foram registradas {n} transações no período analisado.",
]


class RulerBenchmark(BaseBenchmark):
    """
    RULER Benchmark para medir tamanho efetivo de contexto.
    
    Testa recuperação de múltiplas informações distribuídas
    ao longo do contexto em posições variadas.
    """
    
    name = "ruler"
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.fact_templates = FACT_TEMPLATES
        self.distractor_templates = DISTRACTOR_TEMPLATES
    
    def _generate_distractor(self) -> str:
        """Gera um parágrafo distrator."""
        template = self.rng.choice(self.distractor_templates)
        return template.format(n=self.rng.randint(1, 999))
    
    def _generate_fact(self, fact_id: int) -> Tuple[str, str, str]:
        """
        Gera um fato com ID único.
        
        Returns:
            (fact_text, query, expected_value)
        """
        template_info = self.rng.choice(self.fact_templates)
        value = template_info["value_generator"](self.rng)
        
        fact = template_info["template"].format(id=fact_id, value=value)
        query = template_info["query_template"].format(id=fact_id)
        
        return fact, query, value
    
    def _build_context_with_facts(
        self,
        num_facts: int,
        num_distractors: int,
        positions: List[float],  # 0.0 a 1.0
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Constrói contexto com fatos em posições específicas.
        
        Returns:
            (context_text, list of {position, fact, query, expected})
        """
        # Gera distratores
        distractors = [self._generate_distractor() for _ in range(num_distractors)]
        
        # Gera fatos
        facts_info = []
        for i, pos in enumerate(positions[:num_facts]):
            fact, query, expected = self._generate_fact(i + 1)
            facts_info.append({
                "position": pos,
                "fact": fact,
                "query": query,
                "expected": expected,
                "fact_id": i + 1
            })
        
        # Monta contexto intercalando fatos nas posições
        all_segments = distractors.copy()
        
        for fact_info in facts_info:
            insert_idx = int(len(all_segments) * fact_info["position"])
            insert_idx = min(insert_idx, len(all_segments))
            all_segments.insert(insert_idx, fact_info["fact"])
        
        context = "\n\n".join(all_segments)
        return context, facts_info
    
    def generate_test_cases(
        self,
        context_sizes: Optional[List[int]] = None,
        num_facts_per_context: int = 3,
        position_sets: Optional[List[List[float]]] = None,
    ) -> List[TestCase]:
        """
        Gera casos de teste RULER.
        
        Args:
            context_sizes: Tamanhos de contexto em número de distratores
            num_facts_per_context: Quantos fatos inserir por contexto
            position_sets: Conjuntos de posições para os fatos
            
        Returns:
            Lista de TestCases
        """
        if context_sizes is None:
            context_sizes = [10, 25, 50, 100]  # Diferentes tamanhos
        
        if position_sets is None:
            position_sets = [
                [0.1, 0.5, 0.9],     # início, meio, fim
                [0.2, 0.4, 0.6],     # mais concentrado
                [0.05, 0.15, 0.95],  # extremos
                [0.45, 0.50, 0.55],  # tudo no meio (lost in middle)
            ]
        
        test_cases = []
        
        for size in context_sizes:
            for pos_set in position_sets:
                context, facts_info = self._build_context_with_facts(
                    num_facts=num_facts_per_context,
                    num_distractors=size,
                    positions=pos_set
                )
                
                # Cria um caso de teste para CADA fato no contexto
                for fact_info in facts_info:
                    pos_label = "_".join([f"{p:.2f}" for p in pos_set])
                    
                    test_cases.append(TestCase(
                        name=f"ruler_size{size}_pos{pos_label}_fact{fact_info['fact_id']}",
                        context=context,
                        query=fact_info["query"],
                        expected=fact_info["expected"],
                        metadata={
                            "context_size": size,
                            "fact_position": fact_info["position"],
                            "position_set": pos_set,
                            "fact_id": fact_info["fact_id"],
                            "context_chars": len(context),
                        }
                    ))
        
        return test_cases
    
    def evaluate_response(
        self,
        response: str,
        expected: str,
        test_case: TestCase
    ) -> float:
        """
        Avalia recuperação exata ou parcial do valor.
        """
        if not response:
            return 0.0
        
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Match exato
        if expected_lower in response_lower:
            return 1.0
        
        # Match parcial: verifica componentes do valor
        # Ex: "ID-12345" -> verifica se "12345" está presente
        for part in expected_lower.replace("-", " ").replace("_", " ").split():
            if len(part) >= 3 and part in response_lower:
                return 0.5
        
        return 0.0
    
    def compute_effective_context_size(
        self,
        results: List["BenchmarkResult"],
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Calcula o tamanho efetivo de contexto baseado nos resultados.
        
        Args:
            results: Resultados do benchmark
            threshold: Score mínimo para considerar "recuperado"
            
        Returns:
            Dict com métricas de tamanho efetivo
        """
        from collections import defaultdict
        
        scores_by_size = defaultdict(list)
        scores_by_position = defaultdict(list)
        
        for r in results:
            if r.benchmark_name != self.name:
                continue
            
            size = r.details.get("context_size", 0)
            position = r.details.get("fact_position", 0.5)
            
            scores_by_size[size].append(r.score)
            
            # Agrupa posições em buckets
            if position < 0.25:
                pos_bucket = "start"
            elif position < 0.75:
                pos_bucket = "middle"
            else:
                pos_bucket = "end"
            
            scores_by_position[pos_bucket].append(r.score)
        
        # Calcula médias
        avg_by_size = {
            size: sum(scores) / len(scores) 
            for size, scores in scores_by_size.items()
        }
        
        avg_by_position = {
            pos: sum(scores) / len(scores)
            for pos, scores in scores_by_position.items()
        }
        
        # Encontra tamanho efetivo (maior tamanho com score >= threshold)
        effective_size = 0
        for size in sorted(avg_by_size.keys()):
            if avg_by_size[size] >= threshold:
                effective_size = size
        
        return {
            "effective_context_size": effective_size,
            "avg_score_by_size": avg_by_size,
            "avg_score_by_position": avg_by_position,
            "lost_in_middle_effect": avg_by_position.get("start", 0) - avg_by_position.get("middle", 0),
        }
