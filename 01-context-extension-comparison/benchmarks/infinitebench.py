"""
InfiniteBench Benchmark

Testa compreensão de contextos extremamente longos (>100K tokens).
Utiliza a tarefa En.QA: QA factual sobre romances em inglês.

Referência: Zhang et al., 2024 — https://arxiv.org/abs/2402.13718
Dataset: xinrongzhang2022/InfiniteBench (HuggingFace)
"""
import re
import logging
from typing import List, Optional
from .base import BaseBenchmark, TestCase

logger = logging.getLogger(__name__)

# Tarefa padrão: QA factual em inglês sobre livros
DEFAULT_TASK = "En.QA"


class InfiniteBenchBenchmark(BaseBenchmark):
    """
    Benchmark InfiniteBench: QA sobre textos com >100K tokens.

    Foca na tarefa En.QA (Question Answering em inglês sobre livros),
    onde o contexto é o livro completo e a resposta é curta e factual.
    """

    name = "infinitebench"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _load_examples(self, task: str, num_examples: int) -> List[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Instale: pip install datasets")

        try:
            ds = load_dataset(
                "xinrongzhang2022/InfiniteBench",
                name=task,
                split="test",
            )
            examples = list(ds.shuffle(seed=self.seed).select(
                range(min(num_examples, len(ds)))
            ))
            return examples
        except Exception as e:
            logger.warning(f"InfiniteBench {task} indisponível: {e}")
            return []

    def generate_test_cases(
        self,
        task: str = DEFAULT_TASK,
        num_examples: int = 10,
    ) -> List[TestCase]:
        examples = self._load_examples(task, num_examples)
        test_cases = []

        for i, ex in enumerate(examples):
            context = ex.get("context", ex.get("input", ""))
            question = ex.get("input", ex.get("question", ""))
            # When "input" is the question, "context" is the book text
            # Some configs have separate "context" and "input" fields
            if context == question:
                # Same field used for both — skip (malformed example)
                continue

            answer = ex.get("answer", ex.get("output", ""))
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            if not context or not question or not answer:
                continue

            test_cases.append(TestCase(
                name=f"infinitebench_{task.replace('.', '_')}_{i}",
                context=context,
                query=question,
                expected=str(answer).strip(),
                metadata={
                    "task": task,
                    "context_chars": len(context),
                },
            ))

        logger.info(f"InfiniteBench ({task}): {len(test_cases)} casos gerados")
        return test_cases

    def evaluate_response(self, response: str, expected: str, test_case: TestCase) -> float:
        if not response:
            return 0.0
        resp = response.lower().strip()
        exp = expected.lower().strip()

        if exp in resp:
            return 1.0

        # Partial word match
        exp_words = [w for w in re.findall(r'\w+', exp) if len(w) > 1]
        if exp_words:
            matched = sum(1 for w in exp_words if w in resp)
            return matched / len(exp_words)
        return 0.0
