import os
import time
import json
import argparse
from datetime import datetime, timezone
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "my_docs")
EVAL_LOG_PATH = os.environ.get("EVAL_LOG_PATH", "eval_traces.jsonl")


def get_chromadb_client():
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )


def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    client = get_chromadb_client()
    return Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

def is_relevant(source: str, expected_sources: list[str]) -> bool:
    if not source:
        return False
    return any(exp.lower() in source.lower() for exp in expected_sources)

def score_query(retrieved_sources: list[str], expected_sources: list[str]) -> dict:
    hits = [is_relevant(s, expected_sources) for s in retrieved_sources]
    k = len(retrieved_sources)

    num_relevant_retrieved = sum(hits)
    precision_at_k = round(num_relevant_retrieved / k, 3) if k else 0.0

    found_expected = {
        exp for exp in expected_sources
        if any(exp.lower() in s.lower() for s in retrieved_sources)
    }
    recall_at_k = round(len(found_expected) / len(expected_sources), 3) if expected_sources else 0.0
    rr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            rr = round(1 / rank, 3)
            break

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": rr,
        "hit_at_1": bool(hits[0]) if hits else False,
    }

def run_eval(eval_set_path: str, k: int, log_path: str = EVAL_LOG_PATH) -> dict:
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    vector_store = get_vector_store()

    per_query_results = []
    run_id = f"eval_{int(time.time())}"

    for item in eval_set:
        query = item["query"]
        expected_sources = item["expected_sources"]

        start = time.perf_counter()
        results = vector_store.similarity_search_with_score(query, k=k)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        retrieved_sources = [doc.metadata.get("source", "") for doc, _ in results]
        retrieved_scores = [round(float(score), 4) for _, score in results]

        metrics = score_query(retrieved_sources, expected_sources)

        record = {
            "query": query,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "retrieved_scores": retrieved_scores,
            "latency_ms": latency_ms,
            **metrics,
        }
        per_query_results.append(record)

        status = "OK" if metrics["recall_at_k"] == 1.0 else "MISS"
        print(f"[{status}] '{query[:60]}...' "
              f"recall@{k}={metrics['recall_at_k']} "
              f"precision@{k}={metrics['precision_at_k']} "
              f"rr={metrics['reciprocal_rank']}")

    n = len(per_query_results)
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION_NAME,
        "k": k,
        "num_queries": n,
        "avg_recall_at_k": round(sum(r["recall_at_k"] for r in per_query_results) / n, 3),
        "avg_precision_at_k": round(sum(r["precision_at_k"] for r in per_query_results) / n, 3),
        "mrr": round(sum(r["reciprocal_rank"] for r in per_query_results) / n, 3),
        "hit_at_1_rate": round(sum(r["hit_at_1"] for r in per_query_results) / n, 3),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in per_query_results) / n, 2),
        "queries": per_query_results,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", required=True, help="Path to eval_set.json")
    parser.add_argument("--k", type=int, default=4, help="top-k chunks to retrieve per query")
    parser.add_argument("--log-path", default=EVAL_LOG_PATH)
    args = parser.parse_args()

    summary = run_eval(args.eval_set, args.k, args.log_path)

    print("\n--- eval summary ---")
    print(f"queries:            {summary['num_queries']}")
    print(f"avg recall@{summary['k']}:     {summary['avg_recall_at_k']}")
    print(f"avg precision@{summary['k']}:  {summary['avg_precision_at_k']}")
    print(f"MRR:                {summary['mrr']}")
    print(f"hit@1 rate:         {summary['hit_at_1_rate']}")
    print(f"avg latency:        {summary['avg_latency_ms']} ms")
    print(f"\nfull results appended to {args.log_path}")


if __name__ == "__main__":
    main()