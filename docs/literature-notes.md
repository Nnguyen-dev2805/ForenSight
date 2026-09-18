# Literature Notes

> Tài liệu tham khảo để hình thành hypothesis. Literature không tự động quyết định architecture; mọi module mới vẫn phải được biện minh bằng experiment của project.

Towards Universal Fake Image Detectors that Generalize
Across Generative Models (CVPR 2023) – bài mở đầu về
generalization.

Towards Universal Fake Image Detectors That Generalize Across Generative Models

Detail at: Towards Universal Fake Image Detectors that Generalize Across Generative Models
(CVPR 2023)

Phân tích vì sao các detector truyền thống huấn luyện trên GAN thất bại khi gặp diffusion
hoặc autoregressive generator

Các classfier truyền thống học 1 ranh giới bát đối xứng:

Nó nhận diện dấu vết của các loại ảnh giả đã thấy

Mọi thứ không có các dấu vết đó được đẩy vào lớp real

Tác giả sử dụng đặc trưng cố định của CLIP ViT-L/14 và thử với 2 bộ phân loại đơn giản:

Nearest neighbor

Linear probing

Pretraining trên các tập dữ liệu như ImageNet hay việc làm tăng độ đa dạng của tập dữ liệu
giúp model xây dựng representation tốt hơn

CLIP:

CLIP attention

CLIP attribution

CLIP GradCAM

CLIP token visualization

- GenImage (2023) – benchmark quy mô lớn và các bài toán
đánh giá.

Detail at: GenImage

Dataset gồm 2 triệu ảnh được thu thập từ nhiều model GenImage khác nhau

Dữ liệu đa dạng: Mỗi model GenImage có các đặc trưng Gen ảnh khác nhau --> dẫn đến

việc model classifier có thể hiệu quả ở tệp ảnh này nhưng lại kém hiệu quả ở tệp ảnh khác

Bài báo đề xuất 2 phương pháp:

Cross-genrator Classification: huấn luyện ảnh trên 1 generator rồi kiểm tra trên các
generator khác

Degraded image classification: kiểm tra các detector sau khi ảnh bị giảm độ phân giải ,
làm mờ hoặc nén JPEG

- GenDet (2023) – cải thiện khả năng tổng quát bằng teacher–
student.

Detail at: GenDet

Gendet: xem phát hiện ảnh AI dưới góc nhìn anomaly detection thay vì chỉ là binary
classification

Phương pháp sử dụng:

Một feature extrator CLIP được đóng băng

Một mạng teacher

Một mạng student

Một feature augmeter

Một classifier cuối cùng dựa trên độ chênh lệch giữa teacher và student

Mục tiêu huấn luyện:

Với ảnh thật: đầu ra của teacher và student phải gần nhau

Với ảnh giả: đầu ra của teacher và student phải xa nhau

Feature augmenter cố tạo ra các đặc trưng giả khó hơn , buộc student phải học được
ranh giới tổng quát hơn

- FakeInversion (CVPR 2024) – hướng tiếp cận latent space.

CVPR 2024 Open Access Repository

Detail at: FakeInversion (CVPR 2024)

Ghép ảnh thật và ảnh giả vào để tránh việc model học vẹt các đặc trưng style của generator

Khi nhận vào ảnh đầu vào, sử 1 model đê phân tích noise , tần suất và phân phối hình ảnh ,

sau đó tái tạo lại 1 ảnh khác dựa trên các latent noise để tái tạo lại ảnh. Detector nhận ảnh

gốc + noise map + reconstructor làm tín hiệu đầu vào

- Raising the Bar of AI-generated Image Detection with CLIP
(CVPRW 2024) – tận dụng CLIP để tăng khả năng tổng quát và
độ bền.

CVPR 2024 Open Access Repository

Detail at: Raising the Bar of AI-generated Image Detection with CLIP (CVPRW 2024)

mở rộng và tối ưu hướng CLIP-based detector của Ojha et al.

- AIDE (2024) – kết hợp semantic và artifact features để cải thiện
hiệu quả trên benchmark khó.

[2406.19435] A Sanity Check for AI-generated Image Detection

Detail at: AIDE (2024)

- Survey 2024 – đọc cuối cùng để hệ thống hóa toàn bộ lĩnh vực.

A review of deep learning-based approaches for deepfake content detection - Passos - 2024 -
Expert Systems - Wiley Online Library

Detail at: Survey 2024

Hướng phù hợp nhất cho đề tài của bạn

Từ sáu bài báo, một kiến trúc nghiên cứu khả thi là:

1. CLIP ViT làm feature encoder tổng quát.

2. Một nhánh texture/noise patch để tìm artifact cục bộ.

3. Một nhánh diffusion reconstruction để tạo residual map.

4. Fusion classifier dự đoán real/fake.

5. Chọn top-k patch đóng góp lớn nhất.

6. Đưa ảnh, residual map và các patch đó vào VLM.

7. Dùng LLM tạo giải thích có cấu trúc:

Dấu hiệu quan sát được.

Vị trí trong ảnh.

Mức độ tin cậy.

Khả năng có nguyên nhân khác ngoài AI.

Pipeline hệ thống

Ảnh đầu vào
→ CLIP ViT trích xuất đặc trưng toàn cục và theo patch

→ Teacher–Student Anomaly Detection tính độ bất thường của từng patch
→ Texture/Noise Branch phát hiện artifact cục bộ

→ Diffusion Reconstruction Branch tạo ảnh tái dựng và residual map

→ Fusion Module kết hợp các anomaly score
→ Top-k Patch Selection chọn các vùng đáng ngờ nhất

→ VLM phân tích ảnh gốc, residual map và các patch bất thường
→ LLM sinh giải thích có cấu trúc.

Tầng 2: Explainable Fake Image Detection

1. Towards Explainable Fake Image Detection with Multi-Modal
Large Language Models (2025)

Paper page - Towards Explainable Fake Image Detection with Multi-Modal Large Language
Models

Detail at: Towards Explainable Fake Image Detection with Multi-Modal Large Language Models
(2025)

2. AIGI-Holmes (ICCV 2025)

ICCV 2025 Open Access Repository

Detail at: AIGI-Holmes (ICCV 2025)

3. ForenX

[2508.01402] ForenX: Towards ExplainAIGI-Holmes (ICCV 2025)able AI-Generated Image
Detection with Multimodal Large Language Models

Detail at: ForenX

4. Seeing Before ReasoninSeeing Before Reasoningg

[2509.25502] Seeing Before Reasoning: A Unified Framework for Generalizable and Explainable
Fake Image Detection

Detail at: Seeing Before Reasoning

Tầng 3: Explainable AI (XAI)

Grad-CAM

Score - CAM

LIME

SHAP

RISE

Tầng 4: MLLM Reasoning

Tầng 5: Dataset có Explanation

Holmes-Set

ICCV 2025 Open Access Repository

ForgReason

[2508.01402] ForenX: Towards Explainable AI-Generated Image Detection with Multimodal Large

Language Models

ExplainFake-Bench

[2509.25502] Seeing Before Reasoning: A Unified Framework for Generalizable and Explainable

Fake Image Detection

