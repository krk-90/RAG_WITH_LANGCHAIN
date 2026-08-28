
import os
import logging
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Request,UploadFile,File
from pydantic import BaseModel 
from slowapi import Limiter,_rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

from rag import get_rag_chain
from ingest import load_docs, chunk_documents, store_chunks

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_chain =None
limiter = Limiter(key_func=get_remote_address)

UPLOAD_DIR = "uploaded_docs"
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

@asynccontextmanager
async def lifespan(app:FastAPI):
    global rag_chain
    logger.info("building RAG chain ...")
    rag_chain = get_rag_chain()
    app.state.rag_chain = rag_chain
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info("RAG chain is ready.")

    yield {"rag_chain": rag_chain}

    logger.info("shutting down.")

app = FastAPI(title="RAG api",version="1.0.0",lifespan=lifespan)    
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded,_rate_limit_exceeded_handler)

class Queryrequest(BaseModel):
    question:str
    k:int | None = None

class Sourcechunk(BaseModel):
    content:str
    metadata:str

class Queryresponse(BaseModel):
    answer:str
    sources:list[Sourcechunk]

class UploadResponse(BaseModel):
    filename: str
    chunks_stored: int
    message: str

@app.get("/") 
def endpoint():
    return {"message":"your chatbot is ready for chat."}    

@app.get("/status")
def health():
    return {
            "status": "unhealthy" if rag_chain is None else "healthy",
            "model": rag_chain is not None
        }

@app.post("/upload",response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        file.file.close()
    
    try:
        documents = load_docs(save_path)
        chunks = chunk_documents(documents)
        store_chunks(chunks)
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
    
    return UploadResponse(
        filename=file.filename,
        chunks_stored=len(chunks),
        message="File ingested and stored successfully.",
    )

app.state.rag_chain = get_rag_chain()
    
@app.post("/chat",response_model=Queryresponse)
@limiter.limit("10/minute")
async def query(request: Request,req:Queryrequest):
        if not req.question or not req.question.strip():
            raise HTTPException(status_code=400,detail="Question cannot be empty")
        try:
            results = rag_chain.invoke({"input":req.question})
        except Exception as e:
            logger.exception("RAG chain failed.")    
            raise HTTPException(status_code=500,detail=str(e))
        sources = [
            Sourcechunk(content=doc.page_content[:300],metadata=doc.metadata) for doc in results.get("context", [])
        ]
        return Queryresponse(answer=results["answer"],sources=sources)