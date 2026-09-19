from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở kiến thức."

        context_blocks = []
        for i, r in enumerate(results, start=1):
            source = r.get("metadata", {}).get("source", r.get("id", f"doc_{i}"))
            content = r.get("content", "")
            context_blocks.append(f"[{i}] (Nguồn: {source}):\n{content}")
        context_str = "\n\n".join(context_blocks)

        prompt = (
            f"Dưới đây là các đoạn thông tin ngữ cảnh được truy xuất:\n\n"
            f"{context_str}\n\n"
            f"Dựa vào các ngữ cảnh trên, hãy trả lời câu hỏi sau:\n"
            f"Câu hỏi: {question}\n\n"
            f"Lưu ý: Chỉ sử dụng thông tin từ ngữ cảnh trên và trích dẫn số thứ tự nguồn [1], [2] nếu có. "
            f"Nếu không có đủ thông tin để trả lời, hãy thông báo rằng không tìm thấy thông tin."
        )
        return self.llm_fn(prompt)

