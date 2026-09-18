# AGENTS.md

## Project purpose

**ForenSight** là project **học tập + research** về generalizable và explainable
AI-generated image detection.

Ưu tiên theo thứ tự:

1. correctness;
2. experimental clarity;
3. reproducibility;
4. readability;
5. minimal complexity.

Không tối ưu cho tốc độ thêm feature nếu phải đánh đổi các mục trên.

## Source of truth

Trước khi implement, đọc tài liệu liên quan theo thứ tự:

1. `docs/research-problem.md` — research questions, hypotheses, terminology;
2. `docs/architecture.md` — architecture và stage boundaries;
3. `docs/evaluation-protocol.md` — split, metrics, evaluation rules;
4. `docs/roadmap.md` — milestone hiện tại và gate;
5. plan của milestone đang `ACTIVE` trong `docs/plans/`.

Nếu task đụng đến một quyết định đã được ghi lại, đọc record tương ứng trong
`docs/decisions/` trước khi thay đổi.

Không duplicate research/architecture/evaluation detail vào file này. Nếu có
mâu thuẫn, tài liệu chuyên biệt ở trên là source of truth cho domain của nó.

## Working rules

- Chỉ làm milestone đang `ACTIVE`, trừ khi user yêu cầu rõ khác đi.
- Không âm thầm thay architecture, dataset strategy, split hoặc evaluation
  protocol để cải thiện kết quả.
- Thay đổi protocol hoặc research decision phải cập nhật source-of-truth doc và
  ghi rationale trong `docs/decisions/` khi thay đổi đủ lớn.
- Mỗi implementation đáng kể phải phục vụ một research question hoặc một
  prerequisite/gate đã có trong roadmap.
- Ưu tiên baseline nhỏ nhất có thể kiểm tra hypothesis trước khi thêm complexity.

Research workflow mặc định:

`Experiment -> Observed failure -> Hypothesis -> Minimal change -> Re-evaluate`

Không dùng workflow:

`Read paper -> Add module -> Read paper -> Add module`.

## Coding principles

- Prefer plain PyTorch và explicit training/evaluation code.
- Dùng hàm/class nhỏ, tensor shape rõ ở boundary quan trọng và config đơn giản.
- Reuse code hiện có trước khi tạo abstraction mới.
- Ưu tiên stdlib/native solution và dependency đã có trước dependency mới.
- Không thêm framework/abstraction cho nhu cầu giả định trong tương lai.
- Nếu plan chỉ yêu cầu `concat -> MLP`, không tự nâng thành attention,
  Q-Former, Perceiver hoặc fusion phức tạp hơn.
- Không tự thêm Lightning, Hydra phức tạp, dependency injection, registry/factory
  nhiều tầng, plugin/custom framework hoặc distributed abstraction nếu chưa có
  failure/evidence yêu cầu.

## Experiment invariants

- Không tune threshold hoặc hyperparameter bằng test data.
- Split/manifest dùng cho experiment phải reproducible và được version/log rõ.
- Config, seed và threshold source phải có thể truy vết cho result chính.
- Baseline so sánh phải dùng protocol tương đương.
- Không đổi preprocessing sau khi nhìn test result mà không version + rerun các
  baseline liên quan.
- Chi tiết metric, repeated runs và evidence evaluation phải theo
  `docs/evaluation-protocol.md`.

## Research integrity

Luôn phân biệt **implemented**, **tested**, **experimentally validated** và
**hypothesis only**; không nâng mức claim nếu evidence chưa đủ.

Không gọi method là `generalizable`, `faithful`, `evidence-preserving` hoặc
`better` nếu chưa đạt evidence requirement tương ứng trong evaluation protocol.

Terminology về candidate/validated/explanation-grounded evidence lấy từ
`docs/research-problem.md` và `docs/evaluation-protocol.md`; không tự định nghĩa
lại trong code hoặc report.

## Data and artifact safety

- Không commit raw datasets, model checkpoints, generated caches hoặc secrets.
- Không commit API keys, access tokens hoặc credentials dưới bất kỳ hình thức nào.
- Raw dataset được coi là immutable; preprocessing phải tạo derived artifacts.
- Derived manifests/splits/preprocessing phải reproducible từ config hoặc script.
- Không overwrite raw data để "sửa" bias; bias-control preprocessing phải có
  version/rationale riêng.

## Commands

Canonical smoke/unit test:

`python3 -m pytest`

- Repo hiện chưa có canonical executable setup/evaluation commands.
- Không tự invent command rồi ghi như project standard.
- Khi executable code tương ứng được thêm, cập nhật section này với command chính
  xác đã chạy được cho setup và evaluation.
- Trước khi claim completion, chạy verification phù hợp với files đã thay đổi.

## Reporting

Sau mỗi implementation/experiment task, report ngắn gọn:

- **Changed:** thay đổi gì;
- **Why:** phục vụ milestone/RQ nào;
- **Validated:** đã kiểm tra bằng cách nào;
- **Learned:** kết quả cho biết gì;
- **Next:** bước nhỏ tiếp theo.
