import json
import yake
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import numpy as np

# Kết nối với Qdrant Cloud
qdrant_client = QdrantClient(
    url="https://7033a553-e8e8-417b-bf01-3b97a66724bf.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="y4H8XisJLAZfQ700zUIvowvKIRvCiULSxjS2YraI6gGfdGKWpzMZHg",
)

# Nếu chưa có collection, tạo một collection mới
collection_name = "qa_collection"
if collection_name not in [col.name for col in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.DOT)  # Sử dụng Dot Product
    )
    print(f"Collection {collection_name} created.")

# Khởi tạo mô hình NER từ Hugging Face
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Tạo pipeline cho NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Khởi tạo Keyword Extractor (yake)
kw_extractor = yake.KeywordExtractor(lan="vi", top=5)  # Cấu hình YAKE cho từ khóa tiếng Việt

# Hàm để điều chỉnh trọng số embedding dựa trên NER và từ khóa
def adjust_embedding_with_ner_and_keywords(text, embedding_vector):
    # Áp dụng NER để tìm các thực thể quan trọng
    ner_results = ner_pipeline(text)
    important_entities = [ent['word'] for ent in ner_results]

    # Sử dụng YAKE để trích xuất từ khóa
    keywords = kw_extractor.extract_keywords(text)
    important_keywords = [kw[0] for kw in keywords]

    # Kết hợp cả thực thể và từ khóa quan trọng
    important_words = set(important_entities + important_keywords)

    # Tăng trọng số cho các từ khóa và thực thể quan trọng trong embedding
    embedding_vector = np.array(embedding_vector)  # Chuyển đổi sang numpy array
    for word in important_words:
        if word in text:
            embedding_vector *= 1.5  # Tăng trọng số, nhân lên một hệ số
    return embedding_vector

# Hàm để xác định giới tính từ câu hỏi
def determine_gender_from_question(question):
    question = question.lower()
    if any(word in question for word in ["nam", "trai"]):
        return "nam"
    elif any(word in question for word in ["nữ", "gái"]):
        return "nữ"
    return None  # Không có giới tính đề cập

# Hàm để đọc dữ liệu từ file data.txt
def load_questions_and_answers_from_file(file_path):
    qa_pairs = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):  # Đọc từng cặp câu hỏi và câu trả lời
            if i < len(lines) and lines[i].startswith("Q "):  # Câu hỏi
                question = lines[i][2:].strip()  # Bỏ ký tự 'Q ' đầu dòng
                answer = lines[i + 1][2:].strip() if (i + 1) < len(lines) and lines[i + 1].startswith("A ") else "Không có câu trả lời."  # Câu trả lời
                qa_pairs.append((question, answer))
    return qa_pairs

# Load dữ liệu từ file data.txt
file_path = "clean_data.txt"
qa_pairs = load_questions_and_answers_from_file(file_path)

# Load model embeddings
model_name = "BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},  # Sử dụng GPU nếu cần
    encode_kwargs={'normalize_embeddings': False}  # Không chuẩn hóa embedding
)

# Tạo vector embedding cho từng câu hỏi và lưu vào Qdrant
for idx, (question, answer) in enumerate(qa_pairs):
    # Tạo embedding cho câu hỏi
    question_embedding = embeddings.embed_documents([question])[0]
    
    # Điều chỉnh embedding với NER và từ khóa
    adjusted_embedding = adjust_embedding_with_ner_and_keywords(question, question_embedding)

    # Xác định giới tính từ câu hỏi
    gender = determine_gender_from_question(question)

    # Lưu embedding vào Qdrant với payload là câu hỏi và câu trả lời
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[{
            "id": idx,  # Đặt một id duy nhất cho mỗi embedding
            "vector": adjusted_embedding,  # Vector của câu hỏi
            "payload": {
                "question": question,  # Lưu thêm câu hỏi
                "answer": answer,  # Lưu câu trả lời
                "gender": gender  # Lưu thông tin giới tính
            }
        }]
    )

print("All question-answer embeddings uploaded to Qdrant!")
