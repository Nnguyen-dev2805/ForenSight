# Research Decision Log

Thư mục này ghi các quyết định làm thay đổi research protocol hoặc architecture.

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
