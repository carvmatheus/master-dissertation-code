"""
BABILong Benchmark

Testa a capacidade de raciocínio em contextos arbitrariamente longos,
intercalando fatos relevantes dentro de texto distrator (PG19).

Referência: Kuratov et al., 2024 — https://arxiv.org/abs/2406.10149
Dataset: booydar/babilong (HuggingFace)

Contextos disponíveis: 0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 512k, 1M
"""
import re
import logging
from typing import List, Optional
from .base import BaseBenchmark, TestCase

logger = logging.getLogger(__name__)

# Tamanhos de contexto a testar (em tokens aprox.)
DEFAULT_CONTEXT_LENGTHS = ["4k", "8k", "16k", "32k"]

# Tarefas bAbI suportadas pelo BABILong
BABILONG_TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5"]


class BABILongBenchmark(BaseBenchmark):
    """
    Benchmark BABILong: raciocínio sobre fatos dispersos em contexto longo.

    Gera casos de teste em múltiplos tamanhos de contexto para medir
    como a acurácia degrada conforme o contexto cresce além da janela.
    """

    name = "babilong"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _load_examples(
        self, task: str, context_length: str, num_examples: int
    ) -> List[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Instale: pip install datasets")

        try:
            # name = tamanho de contexto (ex: "4k"), split = tarefa (ex: "qa1")
            ds = load_dataset(
                "booydar/babilong",
                name=context_length,
                split=task,
            )
            examples = list(ds.shuffle(seed=self.seed).select(
                range(min(num_examples, len(ds)))
            ))
            return examples
        except Exception as e:
            logger.warning(f"BABILong {task}/{context_length} indisponível: {e}")
            return []

    def generate_test_cases(
        self,
        context_lengths: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        num_examples_per_config: int = 3,
    ) -> List[TestCase]:
        context_lengths = context_lengths or DEFAULT_CONTEXT_LENGTHS
        tasks = tasks or ["qa1", "qa2"]

        test_cases = []
        for length in context_lengths:
            for task in tasks:
                examples = self._load_examples(task, length, num_examples_per_config)
                for i, ex in enumerate(examples):
                    context = ex.get("input", ex.get("context", ""))
                    question = ex.get("question", ex.get("query", ""))
                    answer = ex.get("target", ex.get("answer", ""))
                    if not context or not question or not answer:
                        continue
                    test_cases.append(TestCase(
                        name=f"babilong_{task}_{length}_{i}",
                        context=context,
                        query=question,
                        expected=str(answer).strip(),
                        metadata={
                            "task": task,
                            "context_length": length,
                            "context_chars": len(context),
                        },
                    ))

        logger.info(f"BABILong: {len(test_cases)} casos gerados")
        return test_cases

    def evaluate_response(self, response: str, expected: str, test_case: TestCase) -> float:
        if not response:
            return 0.0
        resp = response.lower().strip()
        exp = expected.lower().strip()

        if exp in resp:
            return 1.0

        # Match parcial por palavras
        exp_words = [w for w in exp.split() if len(w) > 1]
        if exp_words:
            matched = sum(1 for w in exp_words if w in resp)
            return matched / len(exp_words)
        return 0.0
