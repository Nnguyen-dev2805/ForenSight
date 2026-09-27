# Research Decision Log

Thư mục này ghi các quyết định làm thay đổi research protocol hoặc architecture.

## Decision Index

1. [2026-09-18 — Stage 1 Dataset Strategy](2026-09-18-stage1-dataset-strategy.md)
2. [2026-09-18 — Tighten Evaluation and Faithfulness Protocol](2026-09-18-tighten-evaluation-and-faithfulness.md)
3. [2026-09-22 — Adopt Tiny-GenImage as Primary Stage 1 Dataset](2026-09-22-adopt-tiny-genimage.md) — **Superseded** by #4 (the canonical Kaggle dataset contains no SD1.4 split)
4. [2026-09-26 — Adopt Kaggle Tiny-GenImage Seven-Generator Protocol](2026-09-26-adopt-kaggle-tiny-genimage.md)
5. [2026-09-26 — Three Training and Evaluation Protocols on Kaggle Tiny-GenImage](2026-09-26-three-experiment-protocols.md)
6. [2026-09-26 — Align Semantic and Forensic Branch Preprocessing Geometry](2026-09-26-align-branch-preprocessing.md)

Không cần ADR nặng cho mọi chỉnh sửa nhỏ.

Tạo một decision note khi thay đổi một trong các nội dung:

- dataset/split policy;
- primary metric;
- baseline definition;
- stage boundary;
- model/backbone chính;
- freeze/train strategy;
- faithfulness protocol.

Template tối thiểu:

```markdown
# Decision: <title>

Date: YYYY-MM-DD
Status: Accepted | Rejected | Superseded

## Context
Ta quan sát vấn đề gì?

## Evidence
Experiment/kết quả nào dẫn tới quyết định?

## Decision
Ta thay đổi gì?

## Why
Hypothesis/lý do là gì?

## Consequence
Experiment nào cần rerun hoặc docs nào cần cập nhật?
```
