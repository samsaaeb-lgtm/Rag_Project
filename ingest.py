import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks_dir = os.path.join(BASE_DIR, "storage", "app", "medical_processed_chunks")
vector_db_dir = os.path.join(BASE_DIR, "storage", "app", "medical_vector_db")

print("🔍 جاري قراءة الملفات الطبية...")

documents = []

for root, _, files in os.walk(chunks_dir):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    content = item.get("content") or item.get("text") or item.get("page_content", "")
                    metadata = item.get("metadata", {})
                    
                    if not metadata and ("source" in item or "page" in item):
                        metadata = {
                            "source": item.get("source", file),
                            "page": item.get("page", "N/A")
                        }
                    
                    if content:
                        documents.append(Document(page_content=content, metadata=metadata))

total_docs = len(documents)
print(f"✅ تم تحميل {total_docs} قطعة نصية!")

print("⚙️ جاري تحميل موديل الـ Embeddings...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

# معالجة القطع على دفعات صغيرة لتفادي التعليق ورؤية الإنجاز مباشرة
batch_size = 250
print(f"🚀 سيبدأ الآن الحفظ على دفعات (كل دفعة {batch_size} قطعة)...")

vector_store = None

for i in range(0, total_docs, batch_size):
    batch = documents[i:i + batch_size]
    
    if vector_store is None:
        vector_store = Chroma.from_documents(
            documents=batch,
            embedding=embedding_model,
            persist_directory=vector_db_dir
        )
    else:
        vector_store.add_documents(batch)
        
    current_count = min(i + batch_size, total_docs)
    print(f"🔄 تم معالجة وحفظ: {current_count} / {total_docs} قطعة...")

print(f"🎉 تم بناء قاعدة البيانات بنجاح في: {vector_db_dir}")