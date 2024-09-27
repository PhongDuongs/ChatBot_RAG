import json
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Condition

# Kết nối với Qdrant Cloud
qdrant_client = QdrantClient(
    url="https://7033a553-e8e8-417b-bf01-3b97a66724bf.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="y4H8XisJLAZfQ700zUIvowvKIRvCiULSxjS2YraI6gGfdGKWpzMZHg",
)

# Tên collection
collection_name = "qa_collection"

# Load model embeddings
model_name = "BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},  # Sử dụng GPU nếu cần
    encode_kwargs={'normalize_embeddings': False}  # Không chuẩn hóa embedding
)

def calculate_similarity(embedding1, embedding2):
    # Hàm tính toán độ tương đồng giữa hai vector embedding (dot product)
    similarity = sum(a * b for a, b in zip(embedding1, embedding2))
    return similarity

def extract_gender(question):
    # Hàm kiểm tra giới tính trong câu hỏi
    if "nam" in question.lower():
        return "nam"
    elif "nữ" in question.lower():
        return "nữ"
    else:
        return None  # Không xác định giới tính

def find_answer(user_question):
    # Tạo embedding cho câu hỏi của người dùng
    user_embedding = embeddings.embed_documents([user_question])[0]
    
    # Kiểm tra embedding có hợp lệ không
    if user_embedding is None:
        return None, "Không thể tạo embedding cho câu hỏi."

    # Xác định giới tính từ câu hỏi
    gender = extract_gender(user_question)

    # Tạo filter query cho Qdrant dựa trên giới tính
    if gender:
        filter_query = Filter(
            must=[
                FieldCondition(key="gender", match=MatchValue(value=gender))  # Lọc theo giới tính xác định
            ]
        )
    else:
        filter_query = Filter(
            must=[
                FieldCondition(key="gender", match=MatchValue(value=None))  # Lọc các bản ghi có giới tính null
            ]
        )

    # Thực hiện tìm kiếm trong Qdrant, query dựa trên câu hỏi
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=user_embedding,
        limit=1,  # Lấy câu hỏi tương tự nhất
        query_filter=filter_query  # Áp dụng filter giới tính
    )

    # Kiểm tra kết quả tìm kiếm
    if search_result and len(search_result) > 0:
        matched_question = search_result[0].payload.get('question')
        matched_answer = search_result[0].payload.get('answer', "Không có câu trả lời.")
        matched_vector = search_result[0].vector
        
        # Tính toán độ tương đồng giữa câu hỏi của người dùng và câu hỏi trong cơ sở dữ liệu
        similarity_score = calculate_similarity(user_embedding, matched_vector)

        # Nếu độ tương đồng dưới 0.7, tìm kiếm độ tương đồng với câu trả lời
        if similarity_score < 0.7:
            answer_embedding = embeddings.embed_documents([matched_answer])[0]
            answer_similarity = calculate_similarity(user_embedding, answer_embedding)
            return matched_question, matched_answer, answer_similarity
        
        return matched_question, matched_answer, similarity_score
    else:
        return None, "Không tìm thấy câu hỏi tương ứng."


def main():
    print("Hãy nhập câu hỏi của bạn (hoặc nhập 'exit' để thoát): ")
    while True:
        user_question = input("> ")
        if user_question.lower() == 'exit':
            break
        
        matched_question, answer, similarity_score = find_answer(user_question)
        if matched_question:
            print(f"Câu hỏi tương ứng: {matched_question}")
            print(f"Câu trả lời: {answer}")
            print(f"Độ tương đồng: {similarity_score}")
        else:
            print(answer)

if __name__ == "__main__":
    main()
