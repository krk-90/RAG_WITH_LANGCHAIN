import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    score: float


@dataclass
class RAGTrace:
    trace_id: str
    query: str
    started_at: str
    stages: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    prompt: str = ""
    answer: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)


class Timer:
    def __init__(self, trace: RAGTrace, stage_name: str):
        self.trace = trace
        self.stage_name = stage_name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.trace.latency_ms[self.stage_name] = round(elapsed_ms, 2)

def mock_embed(text: str) -> list[float]:
    time.sleep(0.01)
    return [float(len(text) % 7)] * 8


def mock_vector_search(query_vec: list[float], top_k: int = 4) -> list[RetrievedChunk]:
    time.sleep(0.02)
    corpus = [
        ("doc_1", "RAG pipelines combine retrieval with generation to ground answers in source data.", 0.91),
        ("doc_2", "Chunking strategy significantly affects retrieval recall and precision.", 0.84),
        ("doc_3", "Observability tools capture traces of query, retrieval, and generation steps.", 0.79),
        ("doc_4", "Unrelated document about quarterly sales figures.", 0.31),
    ]
    return [RetrievedChunk(doc_id=d, text=t, score=s) for d, t, s in corpus[:top_k]]


def mock_llm_call(prompt: str) -> str:
    time.sleep(0.05)
    return (
        "RAG pipelines improve answer grounding by retrieving relevant chunks "
        "before generation, and observability tools let you trace each stage "
        "to debug retrieval and generation failures separately."
    )


def score_groundedness(answer: str, chunks: list[RetrievedChunk]) -> float:
    context_words = set(" ".join(c.text.lower().split()) for c in chunks)
    context_words = set(" ".join(context_words).split())
    answer_words = set(answer.lower().split())
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return round(len(overlap) / len(answer_words), 3)


def score_retrieval_relevance(chunks: list[RetrievedChunk], threshold: float = 0.5) -> float:
    if not chunks:
        return 0.0
    relevant = [c for c in chunks if c.score >= threshold]
    return round(len(relevant) / len(chunks), 3)

def run_rag_query(query: str, log_path: str = "rag_traces.jsonl") -> RAGTrace:
    trace = RAGTrace(
        trace_id=str(uuid.uuid4()),
        query=query,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    pipeline_start = time.perf_counter()

    with Timer(trace, "embed_query"):
        query_vec = mock_embed(query)

    with Timer(trace, "retrieval"):
        chunks = mock_vector_search(query_vec, top_k=4)
        trace.retrieved_chunks = chunks

    trace.scores["retrieval_relevance"] = score_retrieval_relevance(chunks)

    with Timer(trace, "prompt_construction"):
        context_block = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks)
        prompt = (
            f"Answer the question using only the context below.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        )
        trace.prompt = prompt

    with Timer(trace, "generation"):
        answer = mock_llm_call(prompt)
        trace.answer = answer

    with Timer(trace, "scoring"):
        trace.scores["groundedness"] = score_groundedness(answer, chunks)

    trace.total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

    _log_trace(trace, log_path)
    return trace

def _log_trace(trace: RAGTrace, log_path: str) -> None:
    """Append trace as one JSON line — swap for your observability backend."""
    with open(log_path, "a") as f:
        f.write(json.dumps(asdict(trace)) + "\n")


def summarize_traces(log_path: str = "rag_traces.jsonl") -> dict[str, float]:
    groundedness_scores = []
    relevance_scores = []
    latencies = []

    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            groundedness_scores.append(record["scores"]["groundedness"])
            relevance_scores.append(record["scores"]["retrieval_relevance"])
            latencies.append(record["total_latency_ms"])

    n = len(latencies)
    return {
        "num_queries": n,
        "avg_groundedness": round(sum(groundedness_scores) / n, 3),
        "avg_retrieval_relevance": round(sum(relevance_scores) / n, 3),
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "p95_latency_ms": round(sorted(latencies)[int(n * 0.95)], 2) if n > 1 else latencies[0],
    }

if __name__ == "__main__":
    import os

    log_file = "rag_traces.jsonl"
    if os.path.exists(log_file):
        os.remove(log_file)

    sample_queries = [
        "How does chunking affect RAG retrieval quality?",
        "What does observability give you in a RAG pipeline?",
        "How do you debug a bad RAG answer?",
    ]

    for q in sample_queries:
        trace = run_rag_query(q, log_path=log_file)
        print(f"\nQuery: {q}")
        print(f"  Retrieved {len(trace.retrieved_chunks)} chunks "
              f"(relevance={trace.scores['retrieval_relevance']})")
        print(f"  Groundedness: {trace.scores['groundedness']}")
        print(f"  Latency breakdown: {trace.latency_ms}")
        print(f"  Total latency: {trace.total_latency_ms} ms")

    print("\n--- Aggregate summary across all traces ---")
    print(json.dumps(summarize_traces(log_file), indent=2))