from pathlib import Path
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter , CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache
import traceback

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "kb"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"

# Ensure environment variables from backend/.env are loaded when running utilities
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)



def load_docs_for_creating_chunks(folder_path : str) : 
    docs = [] 
    for file_path in Path(folder_path).iterdir() : 
        if file_path.suffix == ".pdf" : 
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())
        elif file_path.suffix == ".txt" :
            loader = TextLoader(str(file_path) , encoding="utf-8")
            docs.extend(loader.load())
        elif file_path.suffix == ".json" :
            with open(file_path , "r" , encoding="utf-8") as f :
                data = json.load(f)
            content = json.dumps(data , ensure_ascii=False , indent=2)
            docs.append(Document(page_content=content, metadata={"source": str(file_path)}))

    return docs


def get_vector_store_obj(embedding_model : str , docs : list[Document]) :
    embedder = OpenAIEmbeddings(model=embedding_model)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    print(f"Total Chunks : {len(chunks)}")

    vector_store = FAISS.from_documents(chunks ,embedding=embedder)

    return vector_store

def get_retiriever(vector_store_obj) : 
    retriever = vector_store_obj.as_retriever(
        search_type="mmr" , #Maximal Marginal Relevance
        search_kwargs={
            "k": 5  ,  # final docs returned
            "fetch_k": 20 ,# candidate pool before diversity filter
            "lambda_mult": 0.7 # closer to 1 = relevance, closer to 0 = diversity
        }
    )
    return retriever


def retriever_setup(folder_path : str , embedding_model : str) : 
    docs = load_docs_for_creating_chunks(folder_path)
    try:
        embedder = OpenAIEmbeddings(model = embedding_model)
    except Exception as e:
        print("Warning: embeddings initialization failed; returning no-op retriever.")
        print(e)
        traceback.print_exc()
        class NoopRetriever:
            def invoke(self, q):
                return []
        return NoopRetriever()
    
    if Path(FAISS_INDEX_PATH).exists():
        print("Loading FAISS index from disk...")
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings=embedder,
            allow_dangerous_deserialization=True  # safe since YOU saved it
        )
    else:
        # Build from scratch and save
        print("Building FAISS index from documents...")
        docs = load_docs_for_creating_chunks(folder_path)
        vector_store = get_vector_store_obj(embedding_model, docs)
        vector_store.save_local(str(FAISS_INDEX_PATH))  # persist to disk
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")
        
    retriever = get_retiriever(vector_store)

    return retriever



@lru_cache(maxsize=1)
def get_kb_retriever() :
    """ Get a cached retriever obj """
    return retriever_setup(str(KB_PATH) , "text-embedding-3-large")