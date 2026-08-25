import os 
import chromadb
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "my_docs")

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the
provided context. If the answer isn't in the context, say you don't know —
do not make up information.
 
Context:
{context}
 
Question: {input}
 
Answer:
"""

def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
        host = os.environ["CHROMA_HOST"]
    )
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

def get_rag_chain(k:int = 4):
    vectore_store = get_vectorstore()
    retriever =vectore_store.as_retriever(search_kwargs = {"k":k})
    llm_model = ChatOpenAI(
        model=os.environ.get("OPENAI_API_KEY", "gpt-4o-mini"),
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    combine_docs_chain = create_stuff_documents_chain(llm_model,prompt)
    rag_chain = create_retrieval_chain(retriever,combine_docs_chain)
    return rag_chain