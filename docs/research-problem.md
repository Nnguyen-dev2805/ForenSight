# Research Problem

## 1. Bài toán

AI-generated image detector thường gặp hai vấn đề:

1. **Generalization:** detector có thể học artifact/shortcut đặc thù của generator hoặc dataset train và suy giảm mạnh với generator chưa thấy.
2. **Explainability:** MLLM có thể tạo explanation tự nhiên nhưng explanation nghe hợp lý chưa chắc phản ánh đúng tín hiệu model thật sự sử dụng để quyết định `real/fake`.

Project tập trung vào giao điểm của hai vấn đề này.

## 2. Mục tiêu

Xây dựng một hệ thống có khả năng:

- phát hiện ảnh real/fake;
- generalize qua generator/dataset;
- giữ được local forensic evidence;
- truyền evidence vào MLLM;
- sinh explanation dựa trên evidence;
- kiểm tra faithfulness bằng intervention thay vì chỉ đánh giá độ trôi chảy.

## 3. Research questions

### RQ1 — Forensic Perception

**Semantic + forensic representation có giúp detector generalize tốt hơn từng representation riêng không?**

Cần so sánh tối thiểu:

- semantic-only;
- forensic-only;
- semantic + forensic fusion.

Không giả định trước fusion sẽ thắng.

### RQ2 — Evidence Alignment

**Stage 2 có truyền forensic evidence sang MLLM hay chỉ truyền thông tin đủ để đoán nhãn?**

Hai baseline chính:

- **Stage 2A — label-only alignment:** projector học để frozen MLLM dự đoán `real/fake`;
- **Stage 2B — evidence-aware alignment:** projector phải giữ thêm local evidence/location hoặc representation tương ứng.

Token/label accuracy cao ở Stage 2A không đủ để kết luận evidence đã được truyền.

### RQ3 — Explanation Faithfulness

**Explanation của MLLM có thực sự phụ thuộc vào evidence được xác định ở Stage 1/2 không?**

RQ3 chỉ được kiểm tra sau khi có bằng chứng rằng prediction thật sự phụ thuộc vào evidence đó.

Vì vậy project tách hai bước:

1. **Evidence causality diagnostic:** prediction có thay đổi có hệ thống khi evidence bị remove/replace không?
2. **Explanation faithfulness:** sau cùng một intervention, explanation có thay đổi đúng theo evidence còn lại không?

## 4. Evidence terminology

Để tránh over-claim, project dùng ba mức thuật ngữ:

- **Candidate evidence:** patch/token có attribution hoặc contribution score cao.
- **Validated predictive evidence:** candidate evidence đã vượt qua intervention/control test và có ảnh hưởng đo được lên prediction.
- **Explanation-grounded evidence:** validated predictive evidence được explanation phản ánh nhất quán sau intervention.

Không gọi một patch là forensic evidence đã được xác nhận chỉ vì attribution score cao.

## 5. Hypotheses

- **H1:** Semantic và forensic branches chứa tín hiệu bổ sung nhau trên cross-generator evaluation.
- **H2:** Label-only projector có thể giữ classification nhưng làm mất thông tin evidence.
- **H3:** Evidence-aware alignment cải thiện evidence retention mà không làm giảm mạnh detection performance.
- **H4:** Explanation fine-tuning có thể cải thiện answer quality nhưng không đảm bảo faithfulness nếu prediction chưa phụ thuộc đúng vào evidence.

Các hypothesis này có thể bị bác bỏ. Kết quả âm vẫn là kết quả nghiên cứu hợp lệ.

## 6. Scope

### In scope

- binary real/fake detection;
- semantic + forensic visual representations;
- local/patch candidate evidence;
- projector-based MLLM alignment;
- LoRA explanation training;
- cross-generator, cross-dataset, robustness evaluation;
- intervention-based evidence causality và explanation faithfulness diagnostics.

### Out of scope ban đầu

- video/deepfake temporal detection;
- audio;
- attribution chính xác generator nào tạo ảnh;
- provenance/watermark system hoàn chỉnh;
- agent/RAG system;
- nhiều forensic branches cùng lúc;
- production-scale serving.

## 7. Success của project

Project không được đánh giá chỉ bằng một Accuracy cao.

Một kết quả tốt cần trả lời được:

1. model generalize đến đâu;
2. model sử dụng tín hiệu nào;
3. candidate evidence nào thật sự ảnh hưởng prediction;
4. evidence có sống sót qua Stage 2 không;
5. explanation có phản ứng đúng khi evidence thay đổi không;
6. failure modes nằm ở đâu.
