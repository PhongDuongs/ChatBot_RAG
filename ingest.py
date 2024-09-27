from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the PDF file
pdf_path = "20280035_20280022_thesis_2706.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# Load embeddings model
model_name = "BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},  # Use 'cuda' for GPU
    encode_kwargs={'normalize_embeddings': False}
)

print("Embedding model loaded")

# Set up Qdrant vector store connection
qdrant_url = "http://localhost:6333"
collection_name = "thesis_db"

# Create Qdrant index from the document chunks
qdrant = Qdrant.from_documents(
    documents=texts,  # The chunked texts
    embedding=embeddings,
    url=qdrant_url,
    collection_name=collection_name,
    prefer_grpc=False
)

print("Qdrant index created successfully")
