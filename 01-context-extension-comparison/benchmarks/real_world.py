"""
Real-world benchmark adapters backed by local JSONL files.

The downloader in scripts/download_benchmark_datasets.py materializes examples
from Hugging Face datasets into a stable, small schema:

{"id": str, "context": str, "query": str, "answers": [str], "metadata": dict}
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import List

from .base import BaseBenchmark, TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "benchmarks"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _token_overlap(response: str, answers: List[str]) -> float:
    resp = _normalize(response)
    if not resp:
        return 0.0

    for answer in answers:
        ans = _normalize(answer)
        if ans and (ans in resp or resp in ans):
            return 1.0

    best = max(answers, key=len) if answers else ""
    words = [w for w in re.findall(r"\w+", _normalize(best)) if len(w) > 2]
    if not words:
        return 0.0
    return sum(1 for word in words if word in resp) / len(words)


def _qa_f1(response: str, answers: List[str]) -> float:
    """Token F1 usado pelas tarefas de QA do LongBench."""
    prediction = re.findall(r"\w+", _normalize(response))
    best = 0.0
    for answer in answers:
        gold = re.findall(r"\w+", _normalize(answer))
        if not prediction or not gold:
            continue
        common = sum((Counter(prediction) & Counter(gold)).values())
        if common:
            precision = common / len(prediction)
            recall = common / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


class JsonlBenchmark(BaseBenchmark):
    """Base class for benchmarks loaded from data/benchmarks/*.jsonl."""

    dataset_file: str = ""
    default_prompt: str = "Responda com base no contexto."

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

    @property
    def path(self) -> Path:
        return self.data_dir / self.dataset_file

    def _load_rows(self, num_examples: int) -> List[dict]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset local não encontrado: {self.path}. "
                "Rode: python scripts/download_benchmark_datasets.py"
            )

        rows = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(rows) >= num_examples:
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def generate_test_cases(self, num_examples: int = 10) -> List[TestCase]:
        test_cases = []
        for i, row in enumerate(self._load_rows(num_examples)):
            answers = row.get("answers") or [row.get("answer", "")]
            answers = [str(a).strip() for a in answers if str(a).strip()]
            if not row.get("context") or not row.get("query") or not answers:
                continue

            metadata = {
                "dataset_id": row.get("id", f"{self.name}_{i}"),
                "context_chars": len(row["context"]),
                "all_answers": answers,
                **row.get("metadata", {}),
            }
            test_cases.append(
                TestCase(
                    name=f"{self.name}_{i}",
                    context=row["context"],
                    query=row["query"],
                    expected=answers[0],
                    metadata=metadata,
                )
            )
        return test_cases

    def evaluate_response(self, response: str, expected: str, test_case: TestCase) -> float:
        return _token_overlap(response, test_case.metadata.get("all_answers", [expected]))


class ZeroScrollsBenchmark(JsonlBenchmark):
    name = "zeroscrolls"
    dataset_file = "zeroscrolls.jsonl"


class NaturalQuestionsBenchmark(JsonlBenchmark):
    name = "naturalquestions"
    dataset_file = "naturalquestions.jsonl"


class LongBenchBenchmark(JsonlBenchmark):
    """Amostras reais das tarefas QA em inglês do LongBench."""

    name = "longbench"
    dataset_file = "longbench.jsonl"

    def generate_test_cases(self, num_qa_cases: int = 5, **_: object) -> List[TestCase]:
        return super().generate_test_cases(num_examples=num_qa_cases)

    def evaluate_response(self, response: str, expected: str, test_case: TestCase) -> float:
        return _qa_f1(response, test_case.metadata.get("all_answers", [expected]))


class TriviaQABenchmark(JsonlBenchmark):
    name = "triviaqa"
    dataset_file = "triviaqa.jsonl"


class HotpotQABenchmark(JsonlBenchmark):
    name = "hotpotqa"
    dataset_file = "hotpotqa.jsonl"


class MuSiQueBenchmark(JsonlBenchmark):
    name = "musique"
    dataset_file = "musique.jsonl"


class MeetingSummarizationBenchmark(JsonlBenchmark):
    name = "meeting_summarization"
    dataset_file = "meeting_summarization.jsonl"
