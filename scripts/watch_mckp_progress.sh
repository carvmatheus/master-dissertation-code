#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DEFAULT_LOG="${REPO_ROOT}/tests/ollama-local/benchmark_mckp_smoke_v5/runs/llama3.1-8b-instruct-q8_0/ctx-8192/run.log"
LOG_PATH="${1:-${DEFAULT_LOG}}"
WATCH_INTERVAL="${WATCH_INTERVAL:-2}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python do projeto nao encontrado: %s\n' "${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${LOG_PATH}" ]]; then
  printf 'Log do benchmark nao encontrado: %s\n' "${LOG_PATH}" >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/benchmark_progress.py" \
  --log "${LOG_PATH}" \
  --models "llama3.1:8b-instruct-q8_0" \
  --contexts "8192" \
  --benchmarks "longbench,zeroscrolls,naturalquestions,triviaqa,hotpotqa,musique,meeting_summarization" \
  --strategies "raw,mckp,mckp_uniform_control" \
  --watch "${WATCH_INTERVAL}"
