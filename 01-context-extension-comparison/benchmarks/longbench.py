"""
LongBench-inspired Tasks

Adaptação de tarefas do LongBench (Bai et al., 2023) para avaliação
de estratégias de extensão de contexto.

Inclui:
- Question Answering (single-doc e multi-doc)
- Sumarização
- Few-shot learning
- Recuperação de informação

Nota: Esta é uma implementação simplificada focada em testar
as estratégias de contexto, não uma reprodução completa do benchmark.
"""
import random
from typing import List, Dict, Any, Optional
from .base import BaseBenchmark, TestCase


# Documentos sintéticos para QA
SAMPLE_DOCUMENTS = [
    {
        "title": "Relatório Trimestral de Vendas",
        "content": """
O departamento comercial registrou um crescimento de 23% nas vendas do terceiro trimestre
em comparação com o mesmo período do ano anterior. O segmento de produtos digitais
foi o principal responsável por esse resultado, com aumento de 45% na receita.

A região Sudeste manteve a liderança com 52% do faturamento total, seguida pela 
região Sul com 21%. O ticket médio subiu para R$ 347,00, representando um incremento
de 12% em relação ao trimestre anterior.

Os principais desafios identificados foram a logística de última milha e a 
integração de canais online e offline. A meta para o próximo trimestre é 
atingir R$ 15 milhões em vendas totais.
        """,
        "qa_pairs": [
            {
                "question": "Qual foi o crescimento percentual nas vendas do terceiro trimestre?",
                "answer": "23%"
            },
            {
                "question": "Qual região teve a maior participação no faturamento?",
                "answer": "Sudeste"
            },
            {
                "question": "Qual é a meta de vendas para o próximo trimestre?",
                "answer": "R$ 15 milhões"
            }
        ]
    },
    {
        "title": "Manual de Procedimentos de Segurança",
        "content": """
Todos os colaboradores devem seguir os protocolos de segurança estabelecidos pela
empresa. O acesso às áreas restritas requer credencial nível 3 ou superior, 
aprovada pelo gestor direto e pelo departamento de segurança.

Em caso de emergência, siga as rotas de evacuação sinalizadas em verde. Os pontos
de encontro estão localizados no estacionamento B e na praça central. O número
de emergência interno é 5555.

Os visitantes devem estar sempre acompanhados por um funcionário autorizado e 
portar crachá de identificação visível. É proibido fotografar ou gravar nas
dependências da empresa sem autorização prévia da diretoria.

Treinamentos de segurança são obrigatórios semestralmente. O próximo treinamento
está agendado para 15 de abril às 10h no auditório principal.
        """,
        "qa_pairs": [
            {
                "question": "Qual nível de credencial é necessário para acessar áreas restritas?",
                "answer": "nível 3"
            },
            {
                "question": "Qual é o número de emergência interno?",
                "answer": "5555"
            },
            {
                "question": "Quando é o próximo treinamento de segurança?",
                "answer": "15 de abril às 10h"
            }
        ]
    },
    {
        "title": "Especificação Técnica do Produto XYZ-500",
        "content": """
O produto XYZ-500 é um dispositivo de monitoramento industrial de última geração.
Suas principais especificações são:

- Processador: ARM Cortex-M4 a 180 MHz
- Memória RAM: 512 KB SRAM
- Armazenamento: 2 MB Flash + slot para cartão SD até 32 GB
- Conectividade: Wi-Fi 802.11 b/g/n, Bluetooth 5.0, Ethernet 10/100
- Alimentação: 12-48V DC ou PoE
- Temperatura de operação: -20°C a +70°C
- Certificações: CE, FCC, ANATEL

O dispositivo suporta até 64 sensores simultâneos via protocolo Modbus RTU/TCP.
A taxa de amostragem máxima é de 1000 amostras por segundo por canal. A garantia
padrão é de 24 meses, extensível para 36 meses mediante contrato de suporte.
        """,
        "qa_pairs": [
            {
                "question": "Qual processador o XYZ-500 utiliza?",
                "answer": "ARM Cortex-M4"
            },
            {
                "question": "Quantos sensores o dispositivo suporta simultaneamente?",
                "answer": "64"
            },
            {
                "question": "Qual é o período de garantia padrão?",
                "answer": "24 meses"
            }
        ]
    },
]

# Templates para sumarização
SUMMARIZATION_TEXTS = [
    {
        "text": """
A inteligência artificial tem transformado diversos setores da economia global nos últimos anos.
No setor financeiro, algoritmos de machine learning são utilizados para detecção de fraudes,
análise de crédito e recomendações de investimento. Na área da saúde, sistemas de IA auxiliam
no diagnóstico de doenças através da análise de imagens médicas e dados de pacientes.

O varejo adotou chatbots e sistemas de recomendação personalizados para melhorar a experiência
do cliente. A indústria manufatureira implementou robôs inteligentes e sistemas de manutenção
preditiva para otimizar a produção. No setor de transportes, veículos autônomos e sistemas
de logística inteligente prometem revolucionar a mobilidade urbana.

Apesar dos avanços, desafios importantes permanecem. Questões éticas sobre viés algorítmico,
privacidade de dados e impacto no emprego precisam ser endereçadas. Regulamentações específicas
estão sendo desenvolvidas em diversos países para garantir o uso responsável dessas tecnologias.
        """,
        "expected_topics": ["inteligência artificial", "setores", "desafios", "ética"]
    },
]


class LongBenchTasks(BaseBenchmark):
    """
    Benchmark com tarefas inspiradas no LongBench.
    
    Foca em QA multi-documento e sumarização.
    """
    
    name = "longbench"
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.documents = SAMPLE_DOCUMENTS
        self.summarization_texts = SUMMARIZATION_TEXTS
    
    def _combine_documents(
        self,
        docs: List[Dict],
        shuffle: bool = True
    ) -> str:
        """Combina múltiplos documentos em um contexto."""
        if shuffle:
            docs = docs.copy()
            self.rng.shuffle(docs)
        
        parts = []
        for doc in docs:
            parts.append(f"=== {doc['title']} ===\n{doc['content'].strip()}")
        
        return "\n\n".join(parts)
    
    def generate_test_cases(
        self,
        task_types: Optional[List[str]] = None,
        num_qa_cases: int = 5,
        use_multi_doc: bool = True,
    ) -> List[TestCase]:
        """
        Gera casos de teste LongBench.
        
        Args:
            task_types: Tipos de tarefa ["qa", "summarization"]
            num_qa_cases: Número de casos de QA
            use_multi_doc: Se True, combina documentos para QA
            
        Returns:
            Lista de TestCases
        """
        if task_types is None:
            task_types = ["qa", "summarization"]
        
        test_cases = []
        
        # Question Answering
        if "qa" in task_types:
            if use_multi_doc:
                # Multi-document QA
                context = self._combine_documents(self.documents)
                
                all_qa_pairs = []
                for doc in self.documents:
                    for qa in doc["qa_pairs"]:
                        all_qa_pairs.append({
                            "doc_title": doc["title"],
                            **qa
                        })
                
                selected_qa = self.rng.sample(
                    all_qa_pairs, 
                    min(num_qa_cases, len(all_qa_pairs))
                )
                
                for i, qa in enumerate(selected_qa):
                    test_cases.append(TestCase(
                        name=f"longbench_multidoc_qa_{i+1}",
                        context=context,
                        query=qa["question"],
                        expected=qa["answer"],
                        metadata={
                            "task_type": "multi_doc_qa",
                            "source_doc": qa["doc_title"],
                            "context_chars": len(context),
                            "num_documents": len(self.documents),
                        }
                    ))
            else:
                # Single-document QA
                for doc in self.documents:
                    for i, qa in enumerate(doc["qa_pairs"][:2]):  # 2 per doc
                        test_cases.append(TestCase(
                            name=f"longbench_singledoc_qa_{doc['title'][:10]}_{i+1}",
                            context=doc["content"].strip(),
                            query=qa["question"],
                            expected=qa["answer"],
                            metadata={
                                "task_type": "single_doc_qa",
                                "source_doc": doc["title"],
                                "context_chars": len(doc["content"]),
                            }
                        ))
        
        # Summarization
        if "summarization" in task_types:
            for i, item in enumerate(self.summarization_texts):
                test_cases.append(TestCase(
                    name=f"longbench_summarization_{i+1}",
                    context=item["text"].strip(),
                    query="Resuma o texto acima em 2-3 frases, destacando os principais pontos.",
                    expected=" ".join(item["expected_topics"]),  # Tópicos esperados
                    metadata={
                        "task_type": "summarization",
                        "expected_topics": item["expected_topics"],
                        "context_chars": len(item["text"]),
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
        Avalia resposta baseado no tipo de tarefa.
        """
        if not response:
            return 0.0
        
        task_type = test_case.metadata.get("task_type", "qa")
        
        if "summarization" in task_type:
            return self._evaluate_summarization(response, test_case)
        else:
            return self._evaluate_qa(response, expected)
    
    def _evaluate_qa(self, response: str, expected: str) -> float:
        """Avalia resposta de QA."""
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()
        
        if expected_lower in response_lower:
            return 1.0
        
        # Match parcial
        expected_words = [w for w in expected_lower.split() if len(w) > 2]
        if expected_words:
            matched = sum(1 for w in expected_words if w in response_lower)
            return matched / len(expected_words)
        
        return 0.0
    
    def _evaluate_summarization(
        self,
        response: str,
        test_case: TestCase
    ) -> float:
        """Avalia sumarização por presença de tópicos esperados."""
        expected_topics = test_case.metadata.get("expected_topics", [])
        
        if not expected_topics:
            return 0.5  # Não conseguimos avaliar sem tópicos
        
        response_lower = response.lower()
        
        matched = 0
        for topic in expected_topics:
            if topic.lower() in response_lower:
                matched += 1
        
        return matched / len(expected_topics)
