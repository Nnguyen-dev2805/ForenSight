# Decision: Adopt Tiny-GenImage as Primary Stage 1 Dataset

Date: 2026-09-22
Status: Accepted

## Context

Dự án ForenSight nghiên cứu câu hỏi RQ1 về khả năng tổng quát hóa cross-generator của mô hình phát hiện ảnh AI.
Trước đây, Stage 1 dự kiến sử dụng `ENSTA-U2IS/GenImage` gốc (khoảng 2.6 triệu ảnh, >100GB lưu trữ dưới dạng các file zip đa phần bị chia nhỏ).
Việc tải, giải nén và lưu trữ hơn 100GB dữ liệu zip đa phần gây tốn kém tài nguyên đĩa, thời gian tải lâu và gây khó khăn khi chạy thử nghiệm trên máy local cá nhân.

## Evidence

- Tập dữ liệu `TheKernel01/Tiny-GenImage` trên Hugging Face cung cấp 35.000 ảnh (28.000 train + 7.000 val) được subsample đồng đều từ GenImage gốc và đóng gói dạng Apache Parquet (~8.3GB).
- Tập dữ liệu này chứa đầy đủ cả 8 generator của GenImage:
  - `0: Real`
  - `1: ADM`
  - `2: BigGAN`
  - `3: GLIDE`
  - `4: Midjourney`
  - `5: SD14` (Stable Diffusion v1.4)
  - `6: SD15` (Stable Diffusion v1.5)
  - `7: VQDM`
  - `8: Wukong`
- Tỷ lệ Real/Fake cân bằng 50/50, và mỗi generator có khoảng 1.750 ảnh fake trong train và 437 ảnh fake trong validation.

## Decision

1. **Chuyển nguồn tải Stage 1 sang `TheKernel01/Tiny-GenImage`**:
   - Thay thế `ENSTA-U2IS/GenImage` bằng `TheKernel01/Tiny-GenImage` làm dataset chính cho Stage 1.
   - Thư mục lưu trữ: `data/raw/tiny_genimage/`.
2. **Quy tắc phân chia không rò rỉ (Zero Generator Leakage Policy)**:
   - Trong `Tiny-GenImage`, tập `train` nguyên bản chứa cả 8 generator.
   - Để bảo toàn tuyệt đối giá trị khoa học của RQ1:
     - **Train split:** Lọc chỉ lấy ảnh thật (`generator == 0`) và ảnh fake **SD1.4** (`generator == 5`).
     - **Val split:** Lọc chỉ lấy ảnh thật (`generator == 0`) và ảnh fake **SD1.4** (`generator == 5`) từ tập validation để calibrate decision threshold $\tau^*$.
     - **In-domain test:** SD1.4 test split.
     - **Near-OOD test:** SD1.5 (`generator == 6`).
     - **Cross-generator OOD tests:** ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong.
3. **Giữ nguyên định dạng Manifest và Pipeline code**:
   - Sử dụng script trích xuất ảnh từ file Parquet ra thư mục local và sinh file `Manifest` JSONL tiêu chuẩn.
   - Toàn bộ pipeline huấn luyện và đánh giá (`R2ImageDataset`, `train_r2.py`, `evaluate_r2.py`) được giữ nguyên 100%.

## Consequence

- Cập nhật `docs/dataset.md` với thông tin về `Tiny-GenImage`.
- Cập nhật `src/forensight/data/inventory.py` và `data/dataset_inventory.json`.
- Cung cấp script tải và trích xuất `scripts/download_tiny_genimage.py`.
- Dung lượng lưu trữ giảm từ >100GB xuống ~8.3GB, giúp thực nghiệm chạy nhanh và thuận tiện trên mọi môi trường.
