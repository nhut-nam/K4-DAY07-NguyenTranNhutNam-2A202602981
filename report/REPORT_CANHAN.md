# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Trần Nhựt Nam
**MSSV:** 2A202602981
**Nhóm:** Nhóm L3A (Chủ đề: Dịch vụ & Quy định Đại học)
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Khi độ tương tự cosine cao (tiến gần về 1), điều đó chứng tỏ hai vector embedding chỉ về cùng một hướng, thể hiện hai đoạn văn bản có ý nghĩa ngữ nghĩa (semantic meaning) rất tương đồng với nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Học máy
- Câu B: Machine Learning
- Tại sao tương đồng: Cả hai câu đều chỉ cùng một khái niệm chuyên ngành khoa học máy tính (tiếng Việt và tiếng Anh tương đương).

**Ví dụ có độ tương tự THẤP:**
- Câu A: Bạn hãy chú ý làm bài tập
- Câu B: Bạn ăn cơm chưa
- Tại sao khác: Một câu nhắc nhở nhiệm vụ học tập, một câu hỏi thăm sinh hoạt đời thường, hai ngữ cảnh không có sự liên quan về ngữ nghĩa.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity đo góc (hướng) giữa hai vector, giúp so sánh bản chất ngữ nghĩa độc lập với độ dài của đoạn văn. Ngược lại, khoảng cách Euclid đo độ dài tuyệt đối nên câu ngắn và câu dài dù cùng chung ý nghĩa vẫn có thể bị tính là có khoảng cách Euclid rất xa nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> *Đáp án:* 23 chunks

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Với overlap = 100, số lượng chunk sẽ là `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25` chunks (tăng từ 23 lên 25 chunks). Tăng độ chồng chéo giúp giữ lại nhiều ngữ cảnh nối tiếp tại ranh giới giữa các chunk, hạn chế rủi ro thông tin bị cắt đứt giữa chừng, qua đó cải thiện chất lượng retrieval (dù phải chấp nhận tốn thêm bộ nhớ lưu trữ và thời gian tính toán).


---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy lookbehind `(?<=[.!?])(?:\s+|\n+)` để phát hiện điểm kết thúc câu dựa trên các dấu chấm, chấm than, hỏi chấm hoặc xuống dòng mà không làm nuốt mất dấu câu ở cuối câu. Sau khi làm sạch các câu rỗng, hàm gom nhóm `max_sentences_per_chunk` câu lại với nhau bằng `' '.join()`. Xử lý ngoại lệ (edge cases): chuỗi rỗng hoặc chỉ có khoảng trắng lập tức trả về `[]`, chuỗi không có dấu câu trả về một chunk duy nhất; tuy nhiên trường hợp từ viết tắt (như `TS.`, `v.v.`) hoặc số thập phân (`3.14`) vẫn có thể bị xem là ranh giới câu nếu đi kèm khoảng trắng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Cài đặt thuật toán hai chiều theo thứ tự ưu tiên các dấu phân cách `["\n\n", "\n", ". ", " ", ""]`: (1) Chiều đệ quy xuống sâu: cắt theo ranh giới ngữ nghĩa lớn trước, chỉ khi mảnh nào vượt quá `chunk_size` mới tiếp tục gọi đệ quy với danh sách separator con nhỏ hơn; (2) Chiều gom lên (merging): các mảnh nhỏ liền kề sau khi tách được ghép lại với nhau tới sát ngưỡng `chunk_size` để chống việc tạo ra hàng loạt chunk vụn làm suy giảm chất lượng retrieval. Base cases gồm: chuỗi rỗng trả về `[]`; chuỗi có độ dài $\le chunk\_size$ trả về `[current_text]`; và khi hết separators (`remaining_separators == []` hoặc `separator == ""`), thuật toán tự động cắt lát theo từng khối `chunk_size` ký tự (slice fallback).

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ in-memory dưới dạng danh sách các dictionary bản ghi chuẩn hóa gồm `id`, `content`, `metadata` (được copy an toàn và gắn mặc định `doc_id`) cùng vector `embedding` được tạo từ `_embedding_fn`. Khi thực hiện `search`, vector embedding của query được tính toán và đem so khớp độ tương tự cosine với từng chunk qua hàm `compute_similarity()`, sau đó sắp xếp danh sách kết quả theo `score` giảm dần và trích xuất `top_k` kết quả (loại bỏ vector embedding 64/1536 chiều để output gọn gàng).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` bắt buộc phải áp dụng cơ chế lọc trước (pre-filtering): lọc các bản ghi khớp với toàn bộ điều kiện trong `metadata_filter` trước, sau đó mới chạy tìm kiếm tương đồng trên tập ứng viên đã lọc. Nếu lọc sau (post-filtering), top-k có thể bị chiếm hết bởi tài liệu không phù hợp và dẫn đến việc trả về rỗng dù store vẫn có dữ liệu thỏa mãn. `delete_document` lọc bỏ mọi bản ghi có `id == doc_id` hoặc `metadata['doc_id'] == doc_id`, so sánh độ dài trước và sau của store để trả về `True` nếu có chunk bị xóa và `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Thực hiện quy trình RAG chuẩn 3 bước: đầu tiên gọi `store.search(question, top_k)` để truy xuất các chunk liên quan nhất. Sau đó định dạng từng chunk rõ ràng, đánh số thứ tự `[1]`, `[2]`, `[3]` kèm nguồn tài liệu (`source` hoặc `id`) để LLM trích dẫn được nguồn gốc thông tin (Source Traceability). Cuối cùng, đưa ngữ cảnh này vào prompt kèm ràng buộc nghiêm ngặt: chỉ sử dụng thông tin được cung cấp, cấm tự suy đoán/bịa đặt, và xử lý an toàn trường hợp không tìm thấy ngữ cảnh phù hợp bằng câu thông báo trực tiếp thay vì gọi LLM vô ích.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\VinUni\Day7\K4-DAY07-NguyenTranNhutNam-2A202602981
plugins: anyio-4.15.1
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.07s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Học máy là một nhánh của trí tuệ nhân tạo. | Học máy là một nhánh của trí tuệ nhân tạo. | cao | 1.0000 | Đúng |
| 2 | Học máy là một nhánh của trí tuệ nhân tạo. | Machine learning is a subset of artificial intelligence. | cao | 0.0153 | Không |
| 3 | Thư viện cung cấp dịch vụ mượn sách và tài liệu học tập. | Sinh viên có thể đến thư viện để mượn giáo trình và tham khảo tài liệu. | cao | -0.0595 | Không |
| 4 | Thời tiết hôm nay tại Hà Nội rất đẹp và có nắng nhẹ. | Đăng ký học phần trên hệ thống học vụ trực tuyến của trường đại học. | thấp | -0.0283 | Đúng |
| 5 | Quy trình đăng ký môn học và thời hạn nộp học phí của sinh viên. | Hướng dẫn đăng ký học phần và quy định đóng học phí đại học. | cao | -0.0335 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là các cặp câu có ý nghĩa tương đương nhau về mặt ngữ nghĩa (như Cặp 2 dịch nghĩa Việt - Anh hay Cặp 5 cùng nói về đăng ký học phần/học phí) lại có điểm tương đồng thực tế xấp xỉ 0.0 khi chạy với `MockEmbedder`. Điều này phản ánh rõ bản chất của hàm băm (MD5 trong MockEmbedder): thuật toán băm phân tán ngẫu nhiên và phá vỡ hoàn toàn tính tương cận ngữ nghĩa (semantic locality), chỉ các xâu ký tự trùng khớp từng chữ mới có vector tương đồng cao. Để hệ thống biểu diễn và bắt được ý nghĩa tương đồng thực sự của văn bản, bắt buộc phải sử dụng các mô hình Deep Learning Embeddings (như Sentence Transformers hay mô hình của OpenAI/Gemini) được học trên không gian biểu diễn ngữ nghĩa liên tục.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Học phí hệ đại học chính quy đại trà cho khóa 68 trong năm học 2026-2027 là bao nhiêu tiền mỗi tín chỉ? | `neu-tuition-decision-985-2026-2027`: "Khóa 68 (tuyển sinh năm 2026): 880.000 đồng/tín chỉ" | 0.6578 | Có | Theo Quyết định 985, mức thu học phí của khóa 68 là 880.000 đồng/tín chỉ. |
| 2 | Sinh viên IBD@NEU khóa 22, đợt tháng 8/2026, phải nộp học phí trước thời hạn nào và bằng phương thức nào? | `neu-tuition-payment-ibd-2026`: "Phương thức nộp: chuyển khoản... Hạn nộp: hết 17:00 ngày 21/08/2026..." | 0.7777 | Có | Hạn nộp 17h00 ngày 21/08/2026 qua chuyển khoản vào TK 2116678989 tại BIDV Hà Nội. |
| 3 | Sinh viên thuộc diện chính sách cần nộp hồ sơ miễn, giảm học phí đợt 2 năm học 2025-2026 ở đâu và trong thời gian nào? (Filter: `audience="student"`) | `neu-tuition-waiver-2025-2026`: "Nộp từ ngày 02/03/2026 đến hết 20/03/2026 tại Phòng CTCT&QLSV phòng 302 Nhà A1" | 0.7664 | Có | Nộp tại Phòng Công tác chính trị và QLSV (phòng 302 Nhà A1) từ 02/03/2026 đến 20/03/2026. |
| 4 | Theo tài liệu tham khảo về học phí NEU 2026, công thức tham khảo để tính học phí là gì? | `neu-tuition-reference-2026`: "Học phí = số tín chỉ đăng ký × đơn giá mỗi tín chỉ" | 0.7274 | Có | Công thức tham khảo là: Học phí = số tín chỉ đăng ký × đơn giá mỗi tín chỉ. |
| 5 | Trong năm học 2026-2027, chương trình Khoa học dữ liệu và Trí tuệ nhân tạo có mức học phí bao nhiêu? | `neu-tuition-decision-985-2026-2027`: "54 triệu đồng: Data Science, AI" | 0.6726 | Có | Cả chương trình Khoa học dữ liệu (Data Science) và Trí tuệ nhân tạo (AI) đều có học phí 54 triệu đồng/năm. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Việc áp dụng bộ lọc `metadata_filter={"audience": "student"}` ở Câu 3 giúp khoanh vùng chính xác văn bản dành riêng cho sinh viên chính sách, tránh bị nhiễu bởi các tài liệu hướng dẫn cổng chung. Ngoài ra, chiến lược `RecursiveChunker` với `chunk_size=700` giữ được trọn vẹn ngữ cảnh của bảng biểu học phí, giúp trích xuất con số chính xác tuyệt đối.


---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |

