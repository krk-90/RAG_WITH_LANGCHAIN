import os 
import argparse
import chromadb
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

def load_docs(path:str):
    if os.path.isdir(path):
        PDF_Loader = DirectoryLoader(path,glob="**/*.pdf",loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader(path,glob="**/*.txt",loader_cls=TextLoader)
        documents = PDF_Loader.load()+text_loader.load()
    elif path.endswith(".pdf"):
        documents = PyPDFLoader(path).load()
    elif path.endswith(".txt"):
        documents = TextLoader(path).load()
    else:
        raise ValueError(f"unsupported file type:{path}.")

    print(f"loaded {len(documents)} documents from {path}")    
    return documents

def chunk_documents(documents,chunk_size = 800,over_lap = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = over_lap,
    )
    chunks = splitter.split_documents(documents)
    print(f"split into {len(chunks)} chunks.")
    return chunks

def get_chromadb_client():
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )

def store_chunks(chunks):
    embeddings =GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                             api_key=os.environ["GOOGLE_API_KEY"])
    client = get_chromadb_client()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME
    )
    print(f"Stored {len(chunks)} chunks in collection '{COLLECTION_NAME}'")
    return vector_store

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",required=True,help="File or folder to ingest")
    parser.add_argument("--chunk-size",type=int,default=800)
    parser.add_argument("--overlap",type=int,default=150)
    args =parser.parse_args()
    documents = load_docs(args.path)
    chunks = chunk_documents(documents,args.chunk_size,args.overlap)
    store_chunks(chunks)
    print("ingestion completed.")

if __name__ == "__main__":
    main()