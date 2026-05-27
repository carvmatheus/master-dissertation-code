"""
NarrativeQA Benchmark

Testa compreensão de leitura sobre documentos longos (livros e roteiros),
com contextos de 50K–400K tokens.

Referência: Kočiský et al., 2018 — https://arxiv.org/abs/1712.07040
Dataset: deepmind/narrativeqa (HuggingFace)
"""
import re
import logging
from typing import List, Optional
from .base import BaseBenchmark, TestCase

logger = logging.getLogger(__name__)


class NarrativeQABenchmark(BaseBenchmark):
    """
    Benchmark NarrativeQA: QA sobre livros e roteiros inteiros.

    Cada exemplo contém o documento completo como contexto e uma pergunta
    sobre o conteúdo. Respostas são texto livre (duas referências por pergunta).
    """

    name = "narrativeqa"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _load_examples(self, num_examples: int) -> List[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Instale: pip install datasets")

        try:
            ds = load_dataset(
                "deepmind/narrativeqa",
                split="test",
            )
            examples = list(ds.shuffle(seed=self.seed).select(
                range(min(num_examples, len(ds)))
            ))
            return examples
        except Exception as e:
            logger.warning(f"NarrativeQA indisponível: {e}")
            return []

    def generate_test_cases(
        self,
        num_examples: int = 10,
    ) -> List[TestCase]:
        examples = self._load_examples(num_examples)
        test_cases = []

        for i, ex in enumerate(examples):
            # NarrativeQA: document -> summary or full text, question, answers (list)
            doc = ex.get("document", {})
            context = doc.get("text", doc.get("summary", ""))
            question = ex.get("question", {})
            if isinstance(question, dict):
                question = question.get("text", "")
            answers = ex.get("answers", [])
            # answers is a list of dicts with "text" key
            if isinstance(answers, list) and answers:
                if isinstance(answers[0], dict):
                    answer_texts = [a.get("text", "") for a in answers if a.get("text")]
                else:
                    answer_texts = [str(a) for a in answers]
            else:
                answer_texts = []

            if not context or not question or not answer_texts:
                continue

            # Use first answer as primary expected; store all for evaluation
            test_cases.append(TestCase(
                name=f"narrativeqa_{i}",
                context=context,
                query=question,
                expected=answer_texts[0].strip(),
                metadata={
                    "context_chars": len(context),
                    "all_answers": answer_texts,
                    "doc_kind": doc.get("kind", "unknown"),
                },
            ))

        logger.info(f"NarrativeQA: {len(test_cases)} casos gerados")
        return test_cases

    def evaluate_response(self, response: str, expected: str, test_case: TestCase) -> float:
        if not response:
            return 0.0
        resp = response.lower().strip()

        # Check against all reference answers
        all_answers = test_case.metadata.get("all_answers", [expected])
        for ans in all_answers:
            exp = ans.lower().strip()
            if exp in resp or resp in exp:
                return 1.0

        # Partial word match against best answer
        best = max(all_answers, key=len) if all_answers else expected
        exp_words = [w for w in re.findall(r'\w+', best.lower()) if len(w) > 2]
        if exp_words:
            matched = sum(1 for w in exp_words if w in resp)
            return matched / len(exp_words)
        return 0.0
