"use client";

import { useState, useCallback, DragEvent, ChangeEvent } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface CompressResult {
  compressed_context: string;
  original_tokens: number;
  final_tokens: number;
  savings_percentage: number;
  execution_time_ms: number;
  strategy: string;
  budget: number;
}

interface AskResult {
  answer: string;
  model: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

const COST_PER_1M_TOKENS = 5.0;

export default function Home() {
  const [document, setDocument] = useState("");
  const [query, setQuery] = useState("");
  const [budget, setBudget] = useState(4000);
  const [strategy, setStrategy] = useState("MCKP");
  const [loading, setLoading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [result, setResult] = useState<CompressResult | null>(null);
  const [answer, setAnswer] = useState<AskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = async (file: File) => {
    if (file.type !== "text/plain" && !file.name.endsWith(".md") && !file.name.endsWith(".txt")) {
      setError("Por enquanto aceitamos apenas .txt e .md. PDF virá em breve.");
      return;
    }
    const text = await file.text();
    setDocument(text);
    setError(null);
  };

  const onDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const compress = async () => {
    if (!document.trim() || !query.trim()) {
      setError("Preencha o documento e a pergunta.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setAnswer(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/compress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document, query, strategy, budget }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Erro na compressão");
      }
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Erro desconhecido");
    } finally {
      setLoading(false);
    }
  };

  const askLLM = async () => {
    if (!result) return;
    setAsking(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          compressed_context: result.compressed_context,
          query,
          model: "gpt-4o-mini",
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Erro ao consultar LLM");
      }
      setAnswer(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Erro desconhecido");
    } finally {
      setAsking(false);
    }
  };

  const formatCurrency = (tokens: number) => {
    const cost = (tokens / 1_000_000) * COST_PER_1M_TOKENS;
    return cost.toLocaleString("pt-BR", { style: "currency", currency: "USD" });
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-7xl space-y-8">
        {/* Header */}
        <header className="text-center space-y-3">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent sm:text-5xl">
            TokenSaver
          </h1>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Reduza o custo de chamadas a LLMs comprimindo documentos gigantes com as
            estratégias MCKP e Sliding Window da dissertação.
          </p>
        </header>

        {/* Inputs */}
        <section className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-4">
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              className={`relative rounded-2xl border-2 border-dashed p-8 text-center transition-all backdrop-blur-sm ${
                dragOver
                  ? "border-cyan-400 bg-cyan-500/10"
                  : "border-white/10 bg-white/5 hover:bg-white/[0.07]"
              }`}
            >
              <input
                type="file"
                accept=".txt,.md"
                onChange={onFileInputChange}
                className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
              />
              <div className="space-y-2 pointer-events-none">
                <p className="text-lg font-medium text-slate-200">
                  Arraste um arquivo ou clique para upload
                </p>
                <p className="text-sm text-slate-500">.txt ou .md (PDF em breve)</p>
              </div>
            </div>

            <textarea
              value={document}
              onChange={(e) => setDocument(e.target.value)}
              placeholder="Cole o texto do documento aqui..."
              className="h-56 w-full rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-200 placeholder:text-slate-600 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400 resize-none"
            />
          </div>

          <div className="space-y-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm space-y-4">
              <div>
                <label className="mb-1 block text-sm font-medium text-slate-300">
                  Pergunta sobre o documento
                </label>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ex: Quais são as multas rescisórias?"
                  className="w-full rounded-xl border border-white/10 bg-slate-950/50 p-3 text-sm text-slate-200 placeholder:text-slate-600 focus:border-cyan-400 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="mb-1 block text-sm font-medium text-slate-300">
                    Estratégia
                  </label>
                  <select
                    value={strategy}
                    onChange={(e) => setStrategy(e.target.value)}
                    className="w-full rounded-xl border border-white/10 bg-slate-950/50 p-3 text-sm text-slate-200 focus:border-cyan-400 focus:outline-none"
                  >
                    <option value="MCKP">MCKP</option>
                    <option value="SLIDING_WINDOW">Sliding Window</option>
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-sm font-medium text-slate-300">
                    Budget (tokens)
                  </label>
                  <input
                    type="number"
                    value={budget}
                    onChange={(e) => setBudget(Number(e.target.value))}
                    min={100}
                    max={128000}
                    className="w-full rounded-xl border border-white/10 bg-slate-950/50 p-3 text-sm text-slate-200 focus:border-cyan-400 focus:outline-none"
                  />
                </div>
              </div>

              <button
                onClick={compress}
                disabled={loading}
                className="w-full rounded-xl bg-gradient-to-r from-cyan-500 to-violet-600 px-5 py-3 font-semibold text-white shadow-lg shadow-cyan-500/20 transition hover:opacity-90 disabled:opacity-50"
              >
                {loading ? "Comprimindo..." : "Comprimir Contexto"}
              </button>
            </div>

            {error && (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
                {error}
              </div>
            )}
          </div>
        </section>

        {/* Resultado */}
        {result && (
          <section className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="grid gap-6 lg:grid-cols-2">
              {/* Lado A: Sem TokenSaver */}
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-slate-200">
                    Sem TokenSaver
                  </h2>
                  <span className="rounded-full bg-red-500/20 px-3 py-1 text-xs font-medium text-red-300">
                    Custo total
                  </span>
                </div>
                <div className="space-y-4">
                  <div className="flex items-end justify-between">
                    <span className="text-slate-400">Tokens</span>
                    <span className="text-3xl font-bold text-slate-100">
                      {result.original_tokens.toLocaleString("pt-BR")}
                    </span>
                  </div>
                  <div className="flex items-end justify-between">
                    <span className="text-slate-400">Estimativa de custo</span>
                    <span className="text-3xl font-bold text-red-400">
                      {formatCurrency(result.original_tokens)}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500">
                    Baseado em ${COST_PER_1M_TOKENS} por 1M de tokens.
                  </p>
                </div>
              </div>

              {/* Lado B: Com TokenSaver */}
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-slate-200">
                    Com TokenSaver
                  </h2>
                  <span className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-300">
                    Economia {result.savings_percentage.toFixed(1)}%
                  </span>
                </div>
                <div className="space-y-4">
                  <div className="flex items-end justify-between">
                    <span className="text-slate-400">Tokens</span>
                    <span className="text-3xl font-bold text-slate-100">
                      {result.final_tokens.toLocaleString("pt-BR")}
                    </span>
                  </div>
                  <div className="flex items-end justify-between">
                    <span className="text-slate-400">Estimativa de custo</span>
                    <span className="text-3xl font-bold text-emerald-400">
                      {formatCurrency(result.final_tokens)}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500">
                    Estratégia {result.strategy} · executado em{" "}
                    {result.execution_time_ms.toFixed(0)}ms · budget {result.budget} tokens.
                  </p>
                </div>
              </div>
            </div>

            {/* Contexto comprimido */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="font-semibold text-slate-200">Contexto Comprimido</h3>
                <button
                  onClick={askLLM}
                  disabled={asking || !result.compressed_context}
                  className="rounded-lg bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-900 transition hover:bg-white disabled:opacity-50"
                >
                  {asking ? "Consultando LLM..." : "Perguntar ao GPT-4o-mini"}
                </button>
              </div>
              <pre className="max-h-96 overflow-auto rounded-xl bg-slate-950/50 p-4 text-xs leading-relaxed text-slate-300 whitespace-pre-wrap">
                {result.compressed_context || "(contexto vazio)"}
              </pre>
            </div>

            {/* Resposta do LLM */}
            {answer && (
              <div className="rounded-2xl border border-cyan-500/20 bg-cyan-500/5 p-6 backdrop-blur-sm animate-in fade-in slide-in-from-bottom-2 duration-300">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-semibold text-cyan-100">Resposta do LLM</h3>
                  <span className="text-xs text-cyan-300/70">{answer.model}</span>
                </div>
                <p className="whitespace-pre-wrap text-slate-200 leading-relaxed">
                  {answer.answer}
                </p>
                {answer.usage && (
                  <p className="mt-3 text-xs text-slate-500">
                    Tokens: {answer.usage.total_tokens} ({answer.usage.prompt_tokens} prompt +{" "}
                    {answer.usage.completion_tokens} completion)
                  </p>
                )}
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  );
}
