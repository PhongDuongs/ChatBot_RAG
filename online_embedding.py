from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
import yake

# Kết nối với Qdrant Cloud
qdrant_client = QdrantClient(
    url="https://7033a553-e8e8-417b-bf01-3b97a66724bf.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="y4H8XisJLAZfQ700zUIvowvKIRvCiULSxjS2YraI6gGfdGKWpzMZHg",
)

# Kiểm tra xem có collection nào chưa
print(qdrant_client.get_collections())

# Nếu chưa có collection, tạo một collection mới
collection_name = "thesis_db"
if collection_name not in [col.name for col in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.DOT)  # Vector kích thước 1024 từ mô hình BAAI/bge-large-en-v1.5
    )
    print(f"Collection {collection_name} created.")

# Load tài liệu PDF
pdf_path = "20280035_20280022_thesis_2706.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Chia nhỏ tài liệu thành các đoạn nhỏ hơn
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# Load model embeddings
model_name = "BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # Sử dụng GPU nếu cần
    encode_kwargs={'normalize_embeddings': False}  # Không chuẩn hóa embedding
)

# Khởi tạo mô hình NER (spacy) và Keyword Extractor (yake)
nlp = spacy.load("vi_core_news_lg")  # Mô hình NER cho tiếng Việt
kw_extractor = yake.KeywordExtractor(lan="vi", top=5)  # Cấu hình YAKE cho từ khóa tiếng Việt

# Hàm để điều chỉnh trọng số embedding dựa trên NER và từ khóa
def adjust_embedding_with_ner_and_keywords(text, embedding_vector):
    # Áp dụng NER để tìm các thực thể quan trọng
    doc = nlp(text)
    important_entities = [ent.text for ent in doc.ents]

    # Sử dụng YAKE để trích xuất từ khóa
    keywords = kw_extractor.extract_keywords(text)
    important_keywords = [kw[0] for kw in keywords]

    # Kết hợp cả thực thể và từ khóa quan trọng
    important_words = set(important_entities + important_keywords)

    # Tăng trọng số cho các từ khóa và thực thể quan trọng trong embedding
    for word in important_words:
        if word in text:
            embedding_vector *= 1.5  # Tăng trọng số, nhân lên một hệ số
    return embedding_vector

# Tạo vector embedding cho từng đoạn văn bản
for idx, text in enumerate(texts):
    # Tạo embedding cho từng đoạn văn bản
    embedding_vector = embeddings.embed_documents([text.page_content])[0]
    
    # Điều chỉnh embedding dựa trên từ khóa và thực thể
    adjusted_embedding_vector = adjust_embedding_with_ner_and_keywords(text.page_content, embedding_vector)

    # Lưu embedding đã điều chỉnh vào Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": idx,  # Đặt một id duy nhất cho mỗi embedding
                "vector": adjusted_embedding_vector,  # Vector sau khi đã điều chỉnh trọng số
                "payload": {"text": text.page_content}  # Lưu thêm nội dung gốc
            }
        ]
    )

print("All embeddings uploaded to Qdrant with adjusted importance for keywords and entities!")
