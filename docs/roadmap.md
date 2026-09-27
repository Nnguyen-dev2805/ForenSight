# Research Roadmap

## Nguyên tắc

Roadmap định nghĩa **thứ tự câu hỏi nghiên cứu**, không phải hợp đồng bắt buộc phải implement mọi stage.

Nếu một milestone thất bại, dừng và phân tích trước khi đi tiếp.

## R0 — Dataset + Evaluation Protocol

**Status:** ACTIVE

### Goal

Xây nền tảng dữ liệu/evaluation cố định để mọi experiment sau so sánh được.

### Deliverables

- dataset inventory;
- deterministic split manifests;
- Kaggle Tiny-GenImage SD1.5 smoke/pilot/main scale manifests;
- threshold policy;
- leakage/bias checks;
- metric implementation/spec;
- experiment result schema;
- small smoke-test subset.

### Gate

Chỉ qua R1 khi:

1. dummy/baseline prediction chạy qua toàn evaluation pipeline và report đúng;
2. generator-disjoint split được xác nhận không leak;
3. threshold được chọn từ validation, không từ test;
4. các bias dễ thấy về resolution/format/source/duplicates đã được kiểm tra và hiểu;
5. không còn dataset ambiguity nghiêm trọng có thể làm sai kết luận research.
6. GenImage compression/resolution shortcut risk đã được định lượng trước khi chốt preprocessing cho Stage 1.

Plan: `docs/plans/r0-dataset-evaluation.md`.

> [!NOTE]
> **Milestone bookkeeping (2026-09-26).** R0 is still marked `ACTIVE` while the R2
> implementation (Stage-1 detectors, the `single`/`logo`/`all7` protocols, configs, and the
> Kaggle kernels) already exists in the repository; the milestone transition was never
> recorded. Before any R2 result is reported: (a) the R0 gate above needs explicit
> sign-off, and (b) numbers produced by `deploy/kaggle/main.py` must not be compared
> against numbers produced from `src/` (see
> `docs/decisions/2026-09-26-align-branch-preprocessing.md`).

---

## R1 — Base MLLM Baseline

### Question

MLLM gốc phân loại và giải thích ảnh AI tốt đến đâu khi chưa forensic specialization?

### Minimal implementation

- inference only;
- fixed prompt;
- deterministic decoding khi có thể.

### Outputs

- label metrics;
- explanation samples;
- hallucination/error taxonomy.

### Gate

Baseline reproducible trên protocol R0.

---

## R2 — Stage 1: Forensic Perception

**Status:** IMPLEMENTED — not formally `ACTIVE` until the R0 gate is signed off (see note under R0).

### Question

Semantic + forensic representation có giúp cross-generator/cross-dataset generalization không?

### Required ablation

- semantic-only;
- forensic-only;
- concat fusion.

### Minimal implementation

- pretrained semantic encoder;
- một forensic encoder;
- concat + small MLP classifier;
- explicit PyTorch training loop.

### Dataset v1

- train: Kaggle Tiny-GenImage SD1.5;
- in-domain: held-out SD1.5;
- cross-generator OOD: Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN;
- protocol modes: `single` (canonical RQ1: train SD1.5 -> 6 unseen OOD), `logo` (7-fold Leave-One-Generator-Out), `all7` (seen-generator upper-bound benchmark);
- external: GenImage++ + WildRF; Chameleon optional.

### Gate

Hiểu rõ contribution của từng branch. Không cần fusion phải thắng; nếu không thắng phải có failure analysis.

---

## R3 — Candidate Evidence Extraction

### Question

Stage 1 đang dựa vào local candidate evidence nào?

### Minimal implementation

- giữ patch/local features trước pooling;
- contribution score đơn giản và cố định;
- top-k patches + coordinates.

### Gate

- định nghĩa contribution score rõ ràng;
- có thể lưu/visualize candidate evidence ổn định cho evaluation subset;
- chưa gọi candidate evidence là validated predictive evidence.

---

## R4 — Stage 2A: Label-only Alignment

### Question

Frozen MLLM có đọc được representation từ Stage 1 qua một projector đơn giản không?

### Minimal implementation

- Stage 1 frozen;
- MLLM frozen;
- 1–2 layer MLP projector;
- output `real/fake`.

### Gate

Đo được label retention so với Stage 1. Không gọi kết quả này là evidence alignment.

---

## R5 — Stage 2B: Evidence-aware Alignment

### Question

Có thể giữ thêm candidate evidence qua projector thay vì chỉ giữ label information không?

### Minimal implementation

Chỉ thiết kế sau khi R4 cho thấy baseline/gap rõ.

Có thể dùng:

- local evidence tokens;
- coordinates;
- evidence supervision nếu dataset hỗ trợ.

### Gate

Evidence retention tốt hơn label-only baseline mà detection không suy giảm không kiểm soát.

---

## R6 — Evidence Causality Diagnostics

### Question

Prediction có thực sự phụ thuộc vào candidate evidence đã được xếp hạng/chọn không?

### Tests

- matched image intervention;
- deletion + insertion;
- feature/token replacement;
- matched low-score controls.

### Gate

Có converging evidence ở dataset level rằng high-score evidence ảnh hưởng prediction nhiều hơn matched controls.

---

## R7 — Stage 3: Explanation LoRA

### Question

Explanation fine-tuning có tăng chất lượng diễn đạt mà vẫn giữ detection/evidence behavior không?

### Baseline

- Stage 1 frozen;
- Stage 2 frozen;
- MLLM LoRA.

### Ablation sau baseline

- Stage 2 low LR;
- Stage 2 trainable nếu có lý do.

### Gate

Explanation quality cải thiện hoặc ít nhất được đặc trưng rõ mà detection behavior không bị phá không kiểm soát.

---

## R8 — Explanation Faithfulness

### Question

Khi validated predictive evidence bị thay đổi, explanation có thay đổi đúng theo evidence còn lại không?

### Tests

- reuse intervention từ R6;
- explanation consistency;
- evidence mention change;
- hallucination/error analysis.

### Gate

Có dataset-level evidence rằng explanation response phụ thuộc vào validated predictive evidence, không chỉ vài example đẹp.

---

## R9 — Full Evaluation

So sánh tối thiểu:

- Base MLLM;
- semantic-only classifier;
- forensic-only classifier;
- fusion classifier;
- Stage 3-only nếu có;
- Stage 1 + prompt nếu có;
- Stage 2A;
- Stage 2B;
- full system.

Báo cáo theo bốn trục:

1. Detection;
2. Generalization;
3. Explanation quality;
4. Faithfulness.

## Khi nào được thêm module mới?

Chỉ khi có chuỗi bằng chứng:

```text
Observed failure
      ↓
Hypothesis
      ↓
Minimal proposed change
      ↓
Experiment that can falsify hypothesis
```

Không thêm module vì novelty hoặc architecture appeal nếu chưa có failure thực nghiệm cần giải quyết.
