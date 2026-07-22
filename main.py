from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import chromadb
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vector_db_dir = os.path.join(BASE_DIR, "storage", "app", "medical_vector_db")

app = FastAPI()

vector_store = None
llm = None
embedding_model = None

class QueryRequest(BaseModel):
    question: str

# 1️⃣ مسار التشخيص والـ Debugging
@app.get("/debug")
async def debug():
    total_count = 0
    collections_list = []
    
    try:
        client = chromadb.PersistentClient(path=vector_db_dir)
        collections_list = [c.name for c in client.list_collections()]
        
        if vector_store:
            total_count = vector_store._collection.count()
    except Exception as e:
        total_count = f"Error: {str(e)}"
            
    return {
        "vector_db_path": vector_db_dir,
        "path_exists": os.path.exists(vector_db_dir),
        "files_in_path": os.listdir(vector_db_dir) if os.path.exists(vector_db_dir) else "غير موجود",
        "existing_collections": collections_list,
        "total_vectors_in_db": total_count
    }

@app.on_event("startup")
async def startup_event():
    global vector_store, llm, embedding_model
    print("جاري تهيئة نظام RAG...")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    # التحقق هل قاعدة البيانات موجودة على السيرفر أم لا
    if not os.path.exists(vector_db_dir) or not os.listdir(vector_db_dir):
        print("قاعدة البيانات غير موجودة، جاري بنائها تلقائياً من ملفات الـ JSON...")
        
        all_docs = []
        chunks_dir = os.path.join(BASE_DIR, "storage", "app", "medical_processed_chunks")
        
        if os.path.exists(chunks_dir):
            for filename in os.listdir(chunks_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(chunks_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            doc = Document(
                                page_content=item.get("page_content", ""),
                                metadata=item.get("metadata", {})
                            )
                            all_docs.append(doc)
            
            if all_docs:
                print(f"تم تحميل {len(all_docs)} قطعة نصية، جاري إنشاء قاعدة بيانات المتجهات...")
                vector_store = Chroma.from_documents(
                    documents=all_docs,
                    embedding=embedding_model,
                    persist_directory=vector_db_dir
                )
                print("تم بناء قاعدة البيانات وحفظها بنجاح!")
            else:
                print("تحذير: لم يتم العثور على بيانات داخل ملفات الـ JSON!")
        else:
            print("خطأ: مجلد الـ chunks غير موجود!")
    else:
        print("قاعدة البيانات موجودة مسبقاً، جاري تحميلها...")
        vector_store = Chroma(
            persist_directory=vector_db_dir,
            embedding_function=embedding_model
        )
        
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    print("النظام جاهز!")

# 2️⃣ مسار الشرح بالذكاء الاصطناعي للأسئلة الكاملة (/ask)
@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not vector_store or not llm:
        raise HTTPException(status_code=500, detail="لم يتم تحميل قاعدة البيانات أو الموديل بعد")
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    system_prompt = (
        "You are an expert academic medical professor specialized in anesthesia education.\n"
        "Your target audience is medical/nursing students who need clear, educational, and well-structured explanations.\n"
        "Use the following pieces of retrieved medical context to answer the student's question.\n"
        "Strict Rules:\n"
        "1. Rely ONLY on the provided context. Do NOT use your pre-trained knowledge or make up any facts.\n"
        "2. Explain the concepts educationally.\n"
        "3. Use structured formatting: bold key terms, use bullet points, and create clear sub-headings.\n"
        "4. For every major fact, clinical guideline, or dosage, cite the source book and page number from the metadata (e.g., [MILLER, p. 756]).\n"
        "5. If the context doesn't contain the answer, say exactly: 'This specific detail is not mentioned in your reference books.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    docs = retriever.invoke(request.question)
    
    context_text = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source', 'Unknown')}, Page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": request.question})
    
    return {"answer": answer}

# 3️⃣ مسار البحث المباشر للكلمات القصيرة (/search)
@app.post("/search")
async def search_documents(request: QueryRequest):
    if not vector_store:
        raise HTTPException(status_code=500, detail="قاعدة البيانات غير محملة")
        
    docs = vector_store.similarity_search(request.question, k=5)
    
    results = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]
    
    return {"results": results}