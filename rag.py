import os
import time
import json
import argparse
from datetime import datetime, timezone
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__),".env"))

COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "my_docs")
QUERY_LOG_PATH = os.environ.get("QUERY_LOG_PATH", "query_traces.jsonl")

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the
provided context. If the answer isn't in the context, say you don't know —
do not make up information.
 
Context:
{context}
 
Question: {input}
 
Answer:
"""
class StageTimingCallback(BaseCallbackHandler):
    def __init__(self):
        self.timings = {}
        self._retriever_start = None
        self._llm_start = None
 
    def on_retriever_start(self, serialized, query, **kwargs):
        self._retriever_start = time.perf_counter()
 
    def on_retriever_end(self, documents, **kwargs):
        if self._retriever_start is not None:
            self.timings["retrieval_ms"] = round((time.perf_counter() - self._retriever_start) * 1000, 2)
 
    # Chat models emit on_chat_model_start rather than on_llm_start
    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._llm_start = time.perf_counter()
 
    def on_llm_start(self, serialized, prompts, **kwargs):
        self._llm_start = time.perf_counter()
 
    def on_llm_end(self, response, **kwargs):
        if self._llm_start is not None:
            self.timings["generation_ms"] = round((time.perf_counter() - self._llm_start) * 1000, 2)
 
def score_groundedness(answer: str, context_docs: list) -> float:
    context_words = set()
    for d in context_docs:
        context_words.update(d.page_content.lower().split())
    answer_words = set(answer.lower().split())
    if not answer_words:
        return 0.0
    return round(len(answer_words & context_words) / len(answer_words), 3)
 

 
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                              api_key=os.environ["GOOGLE_API_KEY"])
    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

def get_rag_chain(k:int = 4):
    vectore_store = get_vectorstore()
    retriever =vectore_store.as_retriever(search_kwargs = {"k":k})
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0.3,
        api_key=os.environ["GOOGLE_API_KEY"],
        timeout=15,     
        max_retries=2, 
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    combine_docs_chain = create_stuff_documents_chain(llm_model,prompt)
    rag_chain = create_retrieval_chain(retriever,combine_docs_chain)
    return rag_chain

def ask(rag_chain, query: str, log_path: str = QUERY_LOG_PATH) -> dict:
    callback = StageTimingCallback()
    started_at = datetime.now(timezone.utc).isoformat()
    total_start = time.perf_counter()
 
    result = rag_chain.invoke({"input": query}, config={"callbacks": [callback]})
 
    total_ms = round((time.perf_counter() - total_start) * 1000, 2)
    answer = result["answer"]
    context_docs = result.get("context", [])
 
    retrieved = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:200],
        }
        for doc in context_docs
    ]
 
    trace = {
        "query": query,
        "answer": answer,
        "started_at": started_at,
        "retrieved_chunks": retrieved,
        "num_chunks_retrieved": len(context_docs),
        "groundedness": score_groundedness(answer, context_docs),
        "timings_ms": {
            **callback.timings,
            "total_ms": total_ms,
            "unaccounted_ms": round(
                total_ms - sum(callback.timings.values()), 2
            ) if callback.timings else total_ms,
        },
    }
 
    with open(log_path, "a") as f:
        f.write(json.dumps(trace) + "\n")
 
    return trace
 
 
def print_trace(trace: dict):
    print(f"\nQ: {trace['query']}")
    print(f"A: {trace['answer']}")
    print(f"\nretrieved {trace['num_chunks_retrieved']} chunks:")
    for c in trace["retrieved_chunks"]:
        page_info = f" (page {c['page']})" if c["page"] is not None else ""
        print(f"  - {c['source']}{page_info}: {c['snippet']!r}...")
    print(f"\ngroundedness: {trace['groundedness']}")
    print(f"timings: {trace['timings_ms']}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Single question to ask. Omit for interactive mode.")
    parser.add_argument("--k", type=int, default=4, help="number of chunks to retrieve")
    parser.add_argument("--log-path", default=QUERY_LOG_PATH)
    args = parser.parse_args()
 
    rag_chain = get_rag_chain(k=args.k)
 
    if args.query:
        trace = ask(rag_chain, args.query, args.log_path)
        print_trace(trace)
        return
 
    print(f"Interactive mode. Collection: {COLLECTION_NAME}. Ctrl+C to exit.\n")
    while True:
        try:
            query = input("Q: ").strip()
            if not query:
                continue
            trace = ask(rag_chain, query, args.log_path)
            print(f"A: {trace['answer']}")
            print(f"   (groundedness={trace['groundedness']}, "
                  f"retrieval={trace['timings_ms'].get('retrieval_ms')}ms, "
                  f"generation={trace['timings_ms'].get('generation_ms')}ms)\n")
        except KeyboardInterrupt:
            print("\nexiting.")
            break
 
 
if __name__ == "__main__":
    main()