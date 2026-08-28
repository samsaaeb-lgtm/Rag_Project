from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import snapshot_download

# التأكد من جلب مفتاح الـ API بشكل صحيح
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vector_db_dir = os.path.join(BASE_DIR, "storage", "app", "medical_vector_db")

# مستودع قاعدة البيانات الجاهزة على Hugging Face
HF_VECTOR_REPO = "dmdmdk/medical-vector-db"

app = FastAPI()

vector_store = None
llm = None
embedding_model = None

class QueryRequest(BaseModel):
    question: str

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

    # 1. تهيئة نموذج التضمين (Embedding Model)
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    # 2. التحقق من وجود قاعدة البيانات محلياً، وإذا لم تكن موجودة يتم تنزيلها مباشرة من Hugging Face
    if not os.path.exists(vector_db_dir) or not os.listdir(vector_db_dir):
        print("📥 قاعدة البيانات غير موجودة محلياً، جاري تنزيلها من Hugging Face...")
        try:
            os.makedirs(vector_db_dir, exist_ok=True)
            snapshot_download(
                repo_id=HF_VECTOR_REPO,
                repo_type="dataset",
                local_dir=vector_db_dir
            )
            print("✅ تم تنزيل وتحميل قاعدة البيانات الجاهزة بنجاح!")
        except Exception as e:
            print(f"❌ خطأ أثناء تنزيل قاعدة البيانات من Hugging Face: {e}")

    # 3. تحميل قاعدة بيانات Chroma مباشرة
    try:
        print("🔄 جاري ربط قاعدة البيانات بالنظام...")
        vector_store = Chroma(
            persist_directory=vector_db_dir,
            embedding_function=embedding_model
        )
        count = vector_store._collection.count()
        print(f"🎉 تم تحميل قاعدة البيانات بنجاح وتحتوي على {count} عنصر!")
    except Exception as e:
        print(f"❌ خطأ فادح أثناء تهيئة Chroma: {e}")

    # 4. تهيئة نموذج الـ LLM مع تمرير المفتاح صراحة
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

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

    # تم التصحيح هنا لاستخدام source_book و page_number بدلاً من source و page
    context_text = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source_book', 'Unknown')}, Page: {d.metadata.get('page_number', 'N/A')}]\n{d.page_content}"
        for d in docs
    ])

    chain = prompt | llm | StrOutputParser()

try:
    answer = chain.invoke({
        "context": context_text,
        "question": request.question
    })

    return {"answer": answer}

except Exception as e:
    print(f"❌ GEMINI ERROR: {type(e).__name__}: {str(e)}")

    raise HTTPException(
        status_code=500,
        detail=f"Gemini error: {str(e)}"
    )

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