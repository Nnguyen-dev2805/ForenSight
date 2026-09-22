# R0 Plan — Dataset + Evaluation

## 1. Objective

Tạo pipeline dữ liệu và evaluation nhỏ, rõ, reproducible để tất cả milestone sau dùng chung.

R0 không train detector phức tạp.

## 2. Research prerequisite

R0 không trực tiếp trả lời RQ1–RQ3. Nó đảm bảo mọi kết luận sau có protocol đáng tin và dataset không có shortcut/leak hiển nhiên chưa được hiểu.

## 2.1 Dataset strategy đã chốt cho baseline đầu tiên

Chi tiết và bảng đặc tả đầy đủ xem tại [docs/dataset.md](../dataset.md).

- **Active runtime datasets (R0 / R1):**
  - **Primary training source:** GenImage — Stable Diffusion v1.4 (`sd14`).
  - **In-domain validation/test:** GenImage — Stable Diffusion v1.4 (`sd14`).
  - **Near-OOD:** GenImage — Stable Diffusion v1.5 (`sd15`).
  - **Cross-generator OOD:** GenImage — Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN.
  - **Modern external benchmark:** GenImage++ — test-only (FLUX.1, SD3, v.v.).
- **Planned future benchmarks (không nằm trong core runtime R0/R1):**
  - **Real-world external benchmark:** WildRF — test-only.
  - **Optional external benchmark:** Chameleon / AIDE — test-only.

GenImage được chọn vì có 8 generator và có protocol train trên SD v1.4 rồi test cross-generator đã được benchmark gốc sử dụng. Tuy nhiên GenImage có bias về JPEG compression và image size, nên project đo lường và định lượng bias ở Task 0.3 trước khi dùng kết luận research. Raw data được giữ immutable; mọi xử lý giảm bias nếu có sẽ là derived artifact riêng.

## 3. Tasks

### Task 0.1 — Dataset inventory

Ghi rõ với mỗi dataset:

- tên/version/source;
- real source;
- fake generators;
- số lượng ảnh;
- resolution distribution;
- format/compression;
- label mapping;
- license/usage notes nếu cần.

**Output:** machine-readable inventory + short Markdown summary.

Inventory tối thiểu phải cover các nguồn đã chốt ở mục 2.1, nhưng chỉ GenImage SD1.4 cần có dữ liệu local ngay để bắt đầu R0. External benchmarks có thể inventory metadata/download path trước và tải sau.

### Task 0.2 — Define split policy

Tách rõ:

- train;
- validation;
- in-domain test;
- cross-generator test;
- cross-dataset test.

Primary cross-generator split phải generator-disjoint và deterministic.

Generator dùng ở cross-generator test không được leak vào train.

Protocol v1:

```text
train:
  GenImage / SD1.4 / train

validation + in-domain test:
  GenImage / SD1.4 / held-out official split

near-OOD:
  GenImage / SD1.5

cross-generator OOD:
  GenImage / Midjourney
  GenImage / ADM
  GenImage / GLIDE
  GenImage / Wukong
  GenImage / VQDM
  GenImage / BigGAN

external modern:
  GenImage++

external real-world:
  WildRF
```

R0 phải kiểm tra exact duplicate (SHA256) và sample identity giữa các manifest trước khi gọi split là sealed (near-duplicate dHash là optional/deferred check). Chi tiết split policy xem tại [docs/dataset.md](../dataset.md).

Để develop nhanh nhưng vẫn giữ diversity theo 1,000 ImageNet classes, tạo các scale manifest deterministic:

- **smoke:** tối đa 1 real + 1 fake / class từ SD1.4 train;
- **pilot:** tối đa 10 real + 10 fake / class từ SD1.4 train;
- **main:** quyết định sau dataset audit và compute profiling; ưu tiên dùng toàn bộ dữ liệu hợp lệ trong budget thay vì chốt số lượng tùy ý từ trước.

Smoke/pilot chỉ phục vụ development và preliminary experiment; kết luận research chính phải dùng main manifest đã khóa.

**Output:** deterministic split manifests + generator membership summary.

### Task 0.3 — Leakage and bias checks

Kiểm tra tối thiểu:

- exact duplicates (SHA256);
- generator leakage;
- label/source correlation;
- resolution imbalance;
- format/compression imbalance;
- semantic category imbalance nếu metadata hỗ trợ;
- near duplicates (optional/deferred research check).

Riêng GenImage, bắt buộc kiểm tra hai confound đã được literature chỉ ra:

- JPEG/compression distribution giữa real và fake;
- width/height/resolution distribution giữa real và fake.

Không tự động re-encode/resize toàn bộ dataset để "sửa bias" trước khi đo. Đầu tiên phải định lượng bias trên raw data; mọi preprocessing để giảm bias sau đó phải trở thành một experiment/decision có version riêng.

Không chỉ report pass/fail. Cần lưu distribution đủ để hiểu mức độ bias.

**Output:** leakage/bias report (JSON + Markdown). Chi tiết xem [docs/dataset.md](../dataset.md).

### Task 0.4 — Metric and threshold policy

Implement tối thiểu:

- AUROC — primary metric;
- F1;
- Accuracy.

Threshold cho F1/Accuracy phải được chọn từ validation set và không được optimize trên test set.

API nên nhận prediction/target đơn giản, không phụ thuộc model.

**Output:** tested metric module + explicit threshold source/value in result. Chi tiết xem [docs/evaluation.md](../evaluation.md).

### Task 0.5 — Evaluation runner

Input conceptual:

```text
sample_id, label, score/probability, metadata
```

Output:

- overall metrics;
- metrics theo split/generator khi metadata có;
- threshold metadata;
- machine-readable result file. Chi tiết xem [docs/evaluation.md](../evaluation.md).

### Task 0.6 — Repeated-run convention

Định nghĩa trước:

- trained baseline chính: target 3 seeds khi compute cho phép;
- report mean ± std;
- 1-seed run chỉ là preliminary result nếu có stochastic training. Chi tiết xem [docs/evaluation.md](../evaluation.md).

R0 chưa cần train model để thực thi rule này, nhưng result schema phải hỗ trợ `seed`.

### Task 0.7 — Smoke-test protocol

Tạo một small synthetic subset (40 ảnh qua 5 classes) để chạy nhanh end-to-end trong development (< 0.05s).
Pipeline chạy qua lệnh `python scripts/run_smoke_test.py`.

Mục đích chỉ kiểm tra pipeline, không dùng làm evidence research chính.

### Task 0.8 — Reproducibility record

Mỗi run lưu:

- config;
- seed;
- split version;
- threshold source/value;
- metrics;
- timestamp. Chi tiết xem [docs/evaluation.md](../evaluation.md).

## 4. Success criteria

R0 hoàn tất khi:

1. cùng một manifest + prediction cho kết quả metrics deterministic;
2. cross-generator split có rule rõ và không leak generator train/test;
3. validation/test boundary rõ, test không được dùng chọn threshold;
4. evaluation runner chạy được không cần model thật;
5. smoke-test đủ nhanh cho coding agent;
6. result schema hỗ trợ seed và repeated-run aggregation;
7. distribution về resolution/format/source/class/generator đã được kiểm tra;
8. exact duplicate (SHA256) được kiểm tra sạch; near-duplicate là optional/deferred check;
9. không còn dataset ambiguity nghiêm trọng có thể làm sai research conclusion;
10. implementation không mâu thuẫn với [docs/evaluation.md](../evaluation.md).
11. GenImage raw-data compression/resolution bias đã được định lượng trước khi quyết định preprocessing cho Stage 1.
12. smoke/pilot/main manifests có deterministic sampling và giữ class balance ở mức khả thi.

## 5. Failure criteria

Dừng và sửa R0 nếu:

- labels/source mapping không rõ;
- split không reproducible;
- generator train/test bị leak;
- threshold được tune bằng test data;
- có duplicate/leak nghiêm trọng chưa xử lý;
- real/fake có resolution/format/source shortcut lớn chưa được hiểu;
- metrics thay đổi do threshold/config ẩn;
- evaluation code gắn cứng với một model cụ thể.

## 6. Non-goals

Không làm trong R0:

- train CLIP/NPR;
- projector;
- MLLM fine-tuning;
- faithfulness intervention implementation hoàn chỉnh;
- distributed training;
- experiment dashboard.

## 7. Coding style cho R0

Ưu tiên cấu trúc nhỏ:

```text
src/
  data.py
  metrics.py
  evaluate.py

scripts/
  build_splits.py
  check_dataset.py

tests/
  test_metrics.py
  test_splits.py
```

Tên file cuối cùng có thể thay đổi nhẹ theo repo thực tế, nhưng tránh tạo framework mới.
