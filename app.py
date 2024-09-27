from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient


# Load embeddings model
model_name = "BAAI/bge-large-en"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # Use 'cuda' for GPU
    encode_kwargs={'normalize_embeddings': False}
)

# Set up Qdrant vector store connection
qdrant_url = "http://localhost:6333"
collection_name = "thesis_db"

client = QdrantClient(
    url = qdrant_url,
    prefer_grpc = False
)

print(client)

db = Qdrant(
    client = client,
    collection_name = collection_name,
    embeddings = embeddings
)

print(db)

query = "SAM là gì?"

docs = db.similarity_search_with_score(query = query, k=5)

for i in docs:
    doc, score = i
    print({"score" : score, "content" : doc.page_content,"metadata" : doc.metadata})
