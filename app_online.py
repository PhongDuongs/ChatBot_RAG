from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Kết nối với Qdrant Cloud
qdrant_client = QdrantClient(
    url="https://7033a553-e8e8-417b-bf01-3b97a66724bf.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="y4H8XisJLAZfQ700zUIvowvKIRvCiULSxjS2YraI6gGfdGKWpzMZHg",
)

# Tên collection
collection_name = "thesis_db"

# Load model embeddings
model_name = "BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},  # Sử dụng GPU nếu cần
    encode_kwargs={'normalize_embeddings': False}  # Chuẩn hóa embedding
)

# Ví dụ về đoạn văn bản để truy vấn
query_text = "hạn chế của SAM là gì ?"

# Tạo vector embedding cho văn bản truy vấn
query_vector = embeddings.embed_documents([query_text])[0]

# Tìm kiếm vector gần nhất trong Qdrant
search_results = qdrant_client.search(
    collection_name=collection_name,
    query_vector=query_vector,  # Vector của truy vấn
    limit=5,  # Số lượng kết quả trả về
)

# In kết quả tìm kiếm
for result in search_results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score}")
    print(f"Text: {result.payload['text']}")
    print("-----")
