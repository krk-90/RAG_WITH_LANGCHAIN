import os 
import time,json
import argparse
import chromadb
from datetime import datetime,timezone
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "my_docs")
INGEST_LOG_PATH = os.environ.get("INGEST_LOG_PATH", "ingest_traces.jsonl")

class Ingestrace:
    def __init__(self,path:str,chunk_size:int,overlap:int):
        self.run_id = f"ingest_{int(time.time())}"
        self.path = path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.started = datetime.now(timezone.utc).isoformat()
        self.stages = {}
        self.counts = {}
        self.errors = []
        self.chunk_stats = {}

    def record_stage(self,name:str,latency_ms: float):
        self.stages[name] = round(latency_ms, 2)

    def record_error(self, stage: str, source: str, error: Exception):
        self.errors.append({"stage": stage, "source": source, "error": str(error)})
        print(f"  [WARN] {stage} failed for {source}: {error}")
 
    def finish_and_log(self, log_path: str = INGEST_LOG_PATH):
        self.total_latency_ms = round(sum(self.stages.values()), 2)
        record = {
            "run_id": self.run_id,
            "path": self.path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.overlap,
            "started_at": self.started_at,
            "stages_ms": self.stages,
            "total_latency_ms": self.total_latency_ms,
            "counts": self.counts,
            "chunk_stats": self.chunk_stats,
            "errors": self.errors,
            "success": len(self.errors) == 0,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record
 
 
class timed_stage: 
    def __init__(self, trace: Ingestrace, name: str):
        self.trace = trace
        self.name = name
 
    def __enter__(self):
        self._start = time.perf_counter()
        return self
 
    def __exit__(self, *exc):
        self.trace.record_stage(self.name, (time.perf_counter() - self._start) * 1000)


def load_docs(path: str, trace: Ingestrace):
    documents = []
    if os.path.isdir(path):
        for label, glob, loader_cls in [
            ("pdf", "**/*.pdf", PyPDFLoader),
            ("txt", "**/*.txt", TextLoader),
        ]:
            try:
                loaded = DirectoryLoader(path, glob=glob, loader_cls=loader_cls).load()
                documents.extend(loaded)
                trace.counts[f"{label}_files_loaded"] = len(loaded)
            except Exception as e:
                trace.record_error("load_docs", f"{path} ({label})", e)
 
    elif path.endswith(".pdf"):
        try:
            documents = PyPDFLoader(path).load()
            trace.counts["pdf_files_loaded"] = len(documents)
        except Exception as e:
            trace.record_error("load_docs", path, e)
 
    elif path.endswith(".txt"):
        try:
            documents = TextLoader(path).load()
            trace.counts["txt_files_loaded"] = len(documents)
        except Exception as e:
            trace.record_error("load_docs", path, e)
 
    else:
        raise ValueError(f"unsupported file type: {path}.")
 
    trace.counts["documents_loaded"] = len(documents)
    print(f"loaded {len(documents)} documents from {path}")
    return documents

def chunk_documents(documents, trace: Ingestrace, chunk_size=800, over_lap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=over_lap,
    )
    chunks = splitter.split_documents(documents)
 
    lengths = [len(c.page_content) for c in chunks]
    trace.counts["chunks_created"] = len(chunks)
    trace.chunk_stats = {
        "count": len(chunks),
        "avg_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
    }
    print(f"split into {len(chunks)} chunks. avg_len={trace.chunk_stats['avg_len']}")
    return chunks

def get_chromadb_client():
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )

def store_chunks(chunks, trace: Ingestrace):
    if not chunks:
        print("  [WARN] no chunks to store, skipping embed/store step")
        return None
 
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    client = get_chromadb_client()
 
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name=COLLECTION_NAME,
        )
    except Exception as e:
        trace.record_error("store_chunks", COLLECTION_NAME, e)
        raise
 
    trace.counts["chunks_stored"] = len(chunks)
    print(f"Stored {len(chunks)} chunks in collection '{COLLECTION_NAME}'")
    return vector_store

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="File or folder to ingest")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--log-path", default=INGEST_LOG_PATH, help="Where to write ingest trace JSONL")
    args = parser.parse_args()
 
    trace = Ingestrace(args.path, args.chunk_size, args.overlap)
 
    with timed_stage(trace, "load_docs"):
        documents = load_docs(args.path, trace)
 
    with timed_stage(trace, "chunk_documents"):
        chunks = chunk_documents(documents, trace, args.chunk_size, args.overlap)
 
    with timed_stage(trace, "store_chunks"):
        store_chunks(chunks, trace)
 
    record = trace.finish_and_log(args.log_path)
 
    print("\n--- ingestion trace ---")
    print(json.dumps(record, indent=2))
    print(f"\ningestion completed in {record['total_latency_ms']} ms "
          f"({'with errors' if record['errors'] else 'no errors'}).")
 
 
if __name__ == "__main__":
    main()