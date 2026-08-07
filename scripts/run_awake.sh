#!/usr/bin/env bash
# Executa qualquer comando mantendo o Mac e a tela acesos até ele terminar.
#
# O piloto MCKP é longo e o PC pode hibernar no meio, corrompendo a medição de
# latência e interrompendo o Ollama. Este wrapper usa caffeinate para impedir
# suspensão de disco (-m), sistema (-s), inatividade (-i) e display (-d)
# enquanto o processo filho estiver ativo (-w PID).
#
# Uso:
#   ./scripts/run_awake.sh .venv/bin/python scripts/run_ollama_benchmark_matrix.py --models ...
#
# Encerra o comando (Ctrl+C) e o caffeinate cai junto, devolvendo o
# comportamento normal de energia à máquina.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "uso: $0 <comando> [args...]" >&2
  exit 2
fi

"$@" &
child=$!
caffeinate -dimsu -w "$child" &
keeper=$!

trap 'kill "$child" 2>/dev/null || true' INT TERM
wait "$child"
status=$?
kill "$keeper" 2>/dev/null || true
exit "$status"
