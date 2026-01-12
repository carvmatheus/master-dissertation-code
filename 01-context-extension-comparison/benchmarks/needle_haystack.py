"""
Needle-in-a-Haystack Benchmark

Teste que insere um fato específico ("agulha") em posição aleatória 
em um contexto longo ("palheiro"), e o modelo deve recuperá-lo.

Expõe:
- Vieses posicionais (início vs meio vs fim)
- Degradação de atenção
- "Lost in the middle" phenomenon
"""
import random
from typing import List, Dict, Any, Optional
from .base import BaseBenchmark, TestCase


# Fatos para inserir como "agulhas"
DEFAULT_NEEDLES = [
    {
        "fact": "O código secreto para acessar o cofre é 7392.",
        "query": "Qual é o código secreto para acessar o cofre?",
        "expected": "7392"
    },
    {
        "fact": "A reunião foi remarcada para o dia 15 de março às 14h.",
        "query": "Para quando foi remarcada a reunião?",
        "expected": "15 de março às 14h"
    },
    {
        "fact": "O nome do projeto secreto é Operação Fênix.",
        "query": "Qual é o nome do projeto secreto?",
        "expected": "Operação Fênix"
    },
    {
        "fact": "A senha do WiFi do escritório é BlueSky2024!",
        "query": "Qual é a senha do WiFi do escritório?",
        "expected": "BlueSky2024!"
    },
    {
        "fact": "O responsável pelo departamento de inovação é Carlos Mendes.",
        "query": "Quem é o responsável pelo departamento de inovação?",
        "expected": "Carlos Mendes"
    },
]

# Texto de preenchimento (filler) para criar o "palheiro"
FILLER_PARAGRAPHS = [
    "A transformação digital tem sido um tema central nas discussões corporativas dos últimos anos. Empresas de todos os setores buscam adaptar seus processos e modelos de negócio para aproveitar as oportunidades oferecidas pelas novas tecnologias.",
    "O mercado financeiro global passou por significativas mudanças regulatórias após a crise de 2008. Novas regras de compliance e governança foram implementadas para aumentar a transparência e reduzir riscos sistêmicos.",
    "A sustentabilidade corporativa deixou de ser apenas uma questão de responsabilidade social para se tornar um diferencial competitivo. Investidores e consumidores cada vez mais valorizam empresas com práticas ambientais responsáveis.",
    "A gestão de talentos evoluiu significativamente com a adoção de metodologias ágeis e culturas organizacionais mais flexíveis. O trabalho remoto e híbrido se consolidou como realidade permanente em muitas organizações.",
    "A análise de dados se tornou fundamental para a tomada de decisões estratégicas. Ferramentas de business intelligence e machine learning permitem extrair insights valiosos de grandes volumes de informação.",
    "A experiência do cliente é hoje considerada um dos principais fatores de diferenciação no mercado. Empresas investem em omnicanalidade e personalização para atender às expectativas crescentes dos consumidores.",
    "A cibersegurança ganhou destaque com o aumento dos ataques virtuais e vazamentos de dados. Organizações precisam investir continuamente em proteção e conscientização de seus colaboradores.",
    "A inovação aberta se consolidou como estratégia para acelerar o desenvolvimento de novos produtos e serviços. Parcerias com startups e universidades complementam os esforços internos de P&D.",
    "O setor de telecomunicações continua em constante evolução com a expansão das redes 5G e a preparação para o 6G. A conectividade de alta velocidade habilita novos casos de uso em IoT e aplicações industriais.",
    "A inteligência artificial generativa representa uma nova fronteira tecnológica com potencial transformador. Modelos de linguagem avançados estão sendo integrados em diversos processos empresariais.",
]


class NeedleInHaystackBenchmark(BaseBenchmark):
    """
    Benchmark Needle-in-a-Haystack.
    
    Testa a capacidade de recuperar informação específica
    inserida em diferentes posições de um contexto longo.
    """
    
    name = "needle_in_haystack"
    
    def __init__(
        self,
        needles: Optional[List[Dict[str, str]]] = None,
        filler_paragraphs: Optional[List[str]] = None,
        seed: int = 42
    ):
        """
        Args:
            needles: Lista de dicts com {fact, query, expected}
            filler_paragraphs: Parágrafos de preenchimento
            seed: Seed para reprodutibilidade
        """
        self.needles = needles or DEFAULT_NEEDLES
        self.filler_paragraphs = filler_paragraphs or FILLER_PARAGRAPHS
        self.rng = random.Random(seed)
    
    def _generate_haystack(self, num_paragraphs: int) -> List[str]:
        """Gera o palheiro com parágrafos de preenchimento."""
        paragraphs = []
        for _ in range(num_paragraphs):
            p = self.rng.choice(self.filler_paragraphs)
            paragraphs.append(p)
        return paragraphs
    
    def _insert_needle(
        self,
        haystack: List[str],
        needle: str,
        position: str  # "start", "middle", "end", ou float 0-1
    ) -> str:
        """Insere a agulha na posição especificada."""
        if position == "start":
            insert_idx = 0
        elif position == "end":
            insert_idx = len(haystack)
        elif position == "middle":
            insert_idx = len(haystack) // 2
        elif isinstance(position, (int, float)):
            insert_idx = int(len(haystack) * position)
        else:
            insert_idx = self.rng.randint(0, len(haystack))
        
        haystack_copy = haystack.copy()
        haystack_copy.insert(insert_idx, needle)
        
        return "\n\n".join(haystack_copy)
    
    def generate_test_cases(
        self,
        num_paragraphs: int = 20,
        positions: Optional[List[str]] = None,
        num_needles: Optional[int] = None,
    ) -> List[TestCase]:
        """
        Gera casos de teste com agulhas em diferentes posições.
        
        Args:
            num_paragraphs: Tamanho do palheiro (em parágrafos)
            positions: Posições para testar ["start", "middle", "end", 0.25, 0.75]
            num_needles: Quantas agulhas usar (default: todas)
            
        Returns:
            Lista de TestCases
        """
        if positions is None:
            positions = ["start", 0.25, "middle", 0.75, "end"]
        
        needles_to_use = self.needles[:num_needles] if num_needles else self.needles
        test_cases = []
        
        for needle_info in needles_to_use:
            for pos in positions:
                haystack = self._generate_haystack(num_paragraphs)
                context = self._insert_needle(
                    haystack,
                    needle_info["fact"],
                    pos
                )
                
                pos_name = pos if isinstance(pos, str) else f"pos_{pos}"
                
                test_cases.append(TestCase(
                    name=f"needle_{pos_name}_{needle_info['expected'][:10]}",
                    context=context,
                    query=needle_info["query"],
                    expected=needle_info["expected"],
                    metadata={
                        "position": pos,
                        "num_paragraphs": num_paragraphs,
                        "context_chars": len(context),
                        "needle_fact": needle_info["fact"],
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
        Avalia se a resposta contém a informação esperada.
        
        Usa matching flexível (substring case-insensitive).
        """
        if not response:
            return 0.0
        
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Match exato
        if expected_lower in response_lower:
            return 1.0
        
        # Match parcial: verifica palavras-chave
        expected_words = expected_lower.split()
        matched = sum(1 for w in expected_words if w in response_lower)
        
        if len(expected_words) > 0:
            return matched / len(expected_words)
        
        return 0.0
