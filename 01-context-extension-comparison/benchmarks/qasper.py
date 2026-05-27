"""
QASPER Benchmark

Testa QA sobre artigos científicos completos (5K–50K tokens).
Perguntas requerem leitura e raciocínio sobre o corpo completo do paper.

Referência: Dasigi et al., 2021 — https://arxiv.org/abs/2105.03011
Dataset: allenai/qasper (HuggingFace)
"""
import re
import logging
from typing import List, Optional
from .base import BaseBenchmark, TestCase

logger = logging.getLogger(__name__)

# Tipos de resposta QASPER
ANSWERABLE_TYPES = {"abstractive", "extractive"}


class QASPERBenchmark(BaseBenchmark):
    """
    Benchmark QASPER: QA sobre papers científicos completos.

    Filtra apenas perguntas com respostas extrativas ou abstrativas
    (descarta "yes/no" e "unanswerable").
    """

    name = "qasper"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _load_examples(self, num_examples: int) -> List[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Instale: pip install datasets")

        try:
            ds = load_dataset(
                "allenai/qasper",
                split="test",
            )
            examples = list(ds.shuffle(seed=self.seed).select(
                range(min(num_examples * 3, len(ds)))  # extra margin for filtering
            ))
            return examples
        except Exception as e:
            logger.warning(f"QASPER indisponível: {e}")
            return []

    def _build_context(self, ex: dict) -> str:
        """Concatena título + abstract + seções do paper."""
        parts = []
        title = ex.get("title", "")
        if title:
            parts.append(f"Title: {title}")
        abstract = ex.get("abstract", "")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        sections = ex.get("full_text", {})
        if isinstance(sections, dict):
            for heading, paragraphs in sections.items():
                if heading:
                    parts.append(f"\n{heading}")
                if isinstance(paragraphs, list):
                    parts.extend(paragraphs)
                elif isinstance(paragraphs, str):
                    parts.append(paragraphs)
        return "\n\n".join(parts)

    def generate_test_cases(
        self,
        num_examples: int = 10,
    ) -> List[TestCase]:
        raw = self._load_examples(num_examples)
        test_cases = []

        for ex in raw:
            if len(test_cases) >= num_examples:
                break

            context = self._build_context(ex)
            if not context:
                continue

            qas = ex.get("qas", {})
            questions = qas.get("question", []) if isinstance(qas, dict) else []
            answers_list = qas.get("answers", []) if isinstance(qas, dict) else []

            for q_idx, (question, answers_entry) in enumerate(zip(questions, answers_list)):
                if len(test_cases) >= num_examples:
                    break
                if not question:
                    continue

                # answers_entry is a dict with "answer" key containing list of answer dicts
                answer_dicts = answers_entry.get("answer", []) if isinstance(answers_entry, dict) else []
                answer_texts = []
                for a in answer_dicts:
                    if not isinstance(a, dict):
                        continue
                    atype = a.get("answerable_type", a.get("type", ""))
                    if atype not in ANSWERABLE_TYPES:
                        continue
                    free_form = a.get("free_form_answer", "")
                    if free_form:
                        answer_texts.append(free_form)
                    extractive = a.get("extractive_spans", [])
                    if isinstance(extractive, list):
                        answer_texts.extend(extractive)

                if not answer_texts:
                    continue

                i = len(test_cases)
                test_cases.append(TestCase(
                    name=f"qasper_{i}",
                    context=context,
                    query=question,
                    expected=answer_texts[0].strip(),
                    metadata={
                        "context_chars": len(context),
                        "all_answers": answer_texts,
                        "paper_title": ex.get("title", ""),
                    },
                ))

        logger.info(f"QASPER: {len(test_cases)} casos gerados")
        return test_cases

    def evaluate_response(self, response: str, expected: str, test_case: TestCase) -> float:
        if not response:
            return 0.0
        resp = response.lower().strip()

        all_answers = test_case.metadata.get("all_answers", [expected])
        for ans in all_answers:
            exp = ans.lower().strip()
            if exp in resp or resp in exp:
                return 1.0

        # Partial word match against longest answer
        best = max(all_answers, key=len) if all_answers else expected
        exp_words = [w for w in re.findall(r'\w+', best.lower()) if len(w) > 2]
        if exp_words:
            matched = sum(1 for w in exp_words if w in resp)
            return matched / len(exp_words)
        return 0.0
