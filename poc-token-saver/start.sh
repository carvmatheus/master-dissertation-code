#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/backend/.venv"

echo "==> Iniciando TokenSaver PoC"

# Backend
if [ ! -d "$VENV" ]; then
  echo "Criando venv do backend..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -r "$ROOT/backend/requirements.txt"
fi

echo "==> Subindo backend (FastAPI) em http://localhost:8000"
"$VENV/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Frontend
echo "==> Subindo frontend (Next.js) em http://localhost:3000"
cd "$ROOT/frontend"
npm install
npm run dev &
FRONTEND_PID=$!

# Cleanup
cleanup() {
  echo ""
  echo "==> Encerrando processos..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM EXIT

echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Pressione Ctrl+C para encerrar."
wait
