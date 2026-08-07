#!/usr/bin/env python3
"""
Download/materialize benchmark datasets for local Ollama runs.

Output schema per line:
{"id": str, "context": str, "query": str, "answers": [str], "metadata": dict}

Default output: data/benchmarks/*.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import tempfile
import zipfile
from itertools import islice
from pathlib import Path
from typing import Iterable, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "benchmarks"


def load_dataset_compat(dataset_id: str, name: str | None, split: str, streaming: bool = True):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming}
    try:
        if name:
            return load_dataset(dataset_id, name, **kwargs)
        return load_dataset(dataset_id, **kwargs)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        raise RuntimeError(
            f"{dataset_id} ainda usa loading script legado. "
            "Use datasets<4, por exemplo: pip install 'datasets==2.18.0'"
        ) from exc


def write_jsonl(path: Path, rows: Iterable[dict], limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=path.suffix, delete=False
        ) as f:
            temporary_path = Path(f.name)
            for row in rows:
                if count >= limit:
                    break
                if not row.get("context") or not row.get("query") or not row.get("answers"):
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        shutil.copyfile(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return count


def split_zeroscrolls_input(ex: dict) -> tuple[str, str]:
    text = ex["input"]
    doc_start = ex.get("document_start_index")
    doc_end = ex.get("document_end_index")
    query_start = ex.get("query_start_index")
    query_end = ex.get("query_end_index")
    if isinstance(doc_start, int) and isinstance(doc_end, int):
        context = text[doc_start:doc_end].strip()
    else:
        context = text
    if isinstance(query_start, int) and isinstance(query_end, int):
        query = text[query_start:query_end].strip()
    else:
        query = text.split("\n", 1)[0].strip()
    return context, query


def zero_scrolls_task_rows(task: str, limit: int) -> Iterable[dict]:
    """Lê o JSONL publicado no ZIP, sem depender do loading script legado."""
    from huggingface_hub import hf_hub_download

    archive_path = hf_hub_download(
        repo_id="tau/zero_scrolls",
        filename=f"{task}.zip",
        repo_type="dataset",
    )
    with zipfile.ZipFile(archive_path) as archive:
        member = f"{task}/validation.jsonl"
        with archive.open(member) as raw_source:
            source = io.TextIOWrapper(raw_source, encoding="utf-8")
            for line in islice((line for line in source if line.strip()), limit):
                yield json.loads(line)


def zero_scrolls_rows(limit: int) -> Iterable[dict]:
    # QMSum e MuSiQue têm adaptadores próprios; não os duplica neste agregado.
    tasks = ["qasper", "narrative_qa"]
    per_task = max(1, limit // len(tasks) + 1)
    emitted = 0
    for task in tasks:
        for ex in zero_scrolls_task_rows(task, per_task):
            context, query = split_zeroscrolls_input(ex)
            yield {
                "id": f"{task}:{ex.get('id', ex.get('pid', emitted))}",
                "context": context,
                "query": query,
                "answers": [str(ex.get("output", "")).strip()],
                "metadata": {"source": "tau/zero_scrolls", "task": task},
            }
            emitted += 1
            if emitted >= limit:
                return


def musique_rows(limit: int) -> Iterable[dict]:
    for ex in zero_scrolls_task_rows("musique", limit):
        context, query = split_zeroscrolls_input(ex)
        yield {
            "id": str(ex.get("id", ex.get("pid", ""))),
            "context": context,
            "query": query,
            "answers": [str(ex.get("output", "")).strip()],
            "metadata": {"source": "tau/zero_scrolls", "task": "musique"},
        }


def qmsum_rows(limit: int) -> Iterable[dict]:
    for ex in zero_scrolls_task_rows("qmsum", limit):
        context, query = split_zeroscrolls_input(ex)
        yield {
            "id": str(ex.get("id", ex.get("pid", ""))),
            "context": context,
            "query": query,
            "answers": [str(ex.get("output", "")).strip()],
            "metadata": {"source": "tau/zero_scrolls", "task": "qmsum"},
        }


def triviaqa_rows(limit: int) -> Iterable[dict]:
    ds = load_dataset_compat("lucadiliello/triviaqa", None, "validation")
    for ex in islice(ds, limit):
        answers = ex.get("answers") or []
        yield {
            "id": str(ex.get("key", "")),
            "context": ex.get("context", ""),
            "query": ex.get("question", ""),
            "answers": [str(a) for a in answers],
            "metadata": {"source": "lucadiliello/triviaqa"},
        }


def hotpotqa_rows(limit: int) -> Iterable[dict]:
    ds = load_dataset_compat("hotpotqa/hotpot_qa", "distractor", "validation")
    for ex in islice(ds, limit):
        ctx = ex.get("context", {})
        parts = []
        for title, sentences in zip(ctx.get("title", []), ctx.get("sentences", [])):
            paragraph = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            parts.append(f"Title: {title}\n{paragraph}")
        yield {
            "id": str(ex.get("id", "")),
            "context": "\n\n".join(parts),
            "query": ex.get("question", ""),
            "answers": [str(ex.get("answer", "")).strip()],
            "metadata": {
                "source": "hotpotqa/hotpot_qa",
                "type": ex.get("type", ""),
                "level": ex.get("level", ""),
            },
        }


def naturalquestions_rows(limit: int) -> Iterable[dict]:
    ds = load_dataset_compat(
        "google-research-datasets/natural_questions", None, "validation"
    )
    emitted = 0
    for ex in ds:
        document = ex.get("document") or {}
        tokens = document.get("tokens") or {}
        token_texts = tokens.get("token") or []
        is_html = tokens.get("is_html") or [False] * len(token_texts)
        context = " ".join(
            str(token) for token, html in zip(token_texts, is_html) if not html
        )
        annotations = ex.get("annotations") or {}
        short_answers = annotations.get("short_answers") or []
        answers = []
        for spans in short_answers:
            if isinstance(spans, dict):
                starts = spans.get("start_token") or []
                ends = spans.get("end_token") or []
                for start, end in zip(starts, ends):
                    answer = " ".join(token_texts[int(start):int(end)]).strip()
                    if answer and answer not in answers:
                        answers.append(answer)
        if not answers:
            continue
        yield {
            "id": str(ex.get("id", emitted)),
            "context": context,
            "query": str((ex.get("question") or {}).get("text", "")),
            "answers": answers,
            "metadata": {
                "source": "google-research-datasets/natural_questions",
                "document_title": document.get("title", ""),
                "answer_type": "short_answer",
            },
        }
        emitted += 1
        if emitted >= limit:
            return


def longbench_rows(limit: int) -> Iterable[dict]:
    from huggingface_hub import hf_hub_download

    archive_path = hf_hub_download(
        repo_id="zai-org/LongBench",
        filename="data.zip",
        repo_type="dataset",
    )
    tasks = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
        "2wikimqa", "musique",
    ]
    per_task = max(1, limit // len(tasks) + 1)
    emitted = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        for task in tasks:
            suffix = f"/{task}.jsonl"
            member = next(
                (name for name in members if name == f"{task}.jsonl" or name.endswith(suffix)),
                None,
            )
            if member is None:
                raise FileNotFoundError(f"Tarefa {task} não encontrada em {archive_path}")
            with archive.open(member) as raw_source:
                source = io.TextIOWrapper(raw_source, encoding="utf-8")
                examples = (json.loads(line) for line in source if line.strip())
                selected = list(islice(examples, per_task))
            for ex in selected:
                yield {
                    "id": str(ex.get("_id", f"{task}:{emitted}")),
                    "context": str(ex.get("context", "")),
                    "query": str(ex.get("input", "")),
                    "answers": [str(answer) for answer in ex.get("answers", [])],
                    "metadata": {
                        "source": "zai-org/LongBench",
                        "task": task,
                        "language": ex.get("language", "en"),
                        "source_length": ex.get("length"),
                        "metric": "qa_f1",
                    },
                }
                emitted += 1
                if emitted >= limit:
                    return


DOWNLOADERS: dict[str, tuple[str, Callable[[int], Iterable[dict]]]] = {
    "longbench": ("longbench.jsonl", longbench_rows),
    "zeroscrolls": ("zeroscrolls.jsonl", zero_scrolls_rows),
    "naturalquestions": ("naturalquestions.jsonl", naturalquestions_rows),
    "triviaqa": ("triviaqa.jsonl", triviaqa_rows),
    "hotpotqa": ("hotpotqa.jsonl", hotpotqa_rows),
    "musique": ("musique.jsonl", musique_rows),
    "meeting_summarization": ("meeting_summarization.jsonl", qmsum_rows),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materializa datasets de benchmark em JSONL local.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--benchmarks", default="all", help="Lista separada por vírgula ou 'all'.")
    parser.add_argument("--limit", type=int, default=50, help="Exemplos por benchmark.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    selected = list(DOWNLOADERS) if args.benchmarks == "all" else [
        item.strip() for item in args.benchmarks.split(",") if item.strip()
    ]

    for name in selected:
        if name not in DOWNLOADERS:
            raise ValueError(f"Benchmark desconhecido: {name}. Opções: {', '.join(DOWNLOADERS)}")
        filename, loader = DOWNLOADERS[name]
        path = output_dir / filename
        print(f">> {name}: baixando/materializando em {path}")
        count = write_jsonl(path, loader(args.limit), args.limit)
        print(f"   {count} exemplos")


if __name__ == "__main__":
    main()
