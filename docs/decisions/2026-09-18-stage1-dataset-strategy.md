# Decision: Stage 1 dataset strategy v1

Date: 2026-09-18
Status: Accepted

## Context

Project cần một dataset strategy đủ nghiêm túc cho research nhưng không làm baseline đầu tiên quá phức tạp hoặc tốn compute. Mục tiêu chính của Stage 1 là đo cross-generator generalization trước khi mở rộng kiến trúc.

## Evidence

- GenImage cung cấp 8 generator và benchmark gốc có protocol train trên SD v1.4 rồi đánh giá trên các generator còn lại.
- GenImage++ là benchmark test-only cho generator hiện đại như FLUX.1 và SD3, nên phù hợp làm external modern generalization test thay vì training source.
- WildRF cung cấp setting real-world/social-platform để bổ sung cho benchmark synthetic có kiểm soát.
- Literature về Unbiased GenImage chỉ ra JPEG compression và image-size bias trong GenImage có thể trở thành shortcut; vì vậy raw dataset phải được audit trước khi tin detection score.
- Chameleon hữu ích cho real-world/human-hard evaluation nhưng yêu cầu xin quyền truy cập, nên giữ optional để không block project.

## Decision

### Primary Stage 1 data

Train baseline đầu tiên trên **GenImage SD1.4**.

Evaluation hierarchy:

1. **In-domain:** SD1.4 held-out split.
2. **Near-OOD:** SD1.5.
3. **Cross-generator OOD:** Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN.
4. **Modern external:** GenImage++.
5. **Real-world external:** WildRF.
6. **Optional:** Chameleon nếu access có sẵn.

### Development scale

Không tải/train toàn bộ benchmark ngay để debug pipeline.

- smoke: tối đa 1 real + 1 fake / ImageNet class;
- pilot: tối đa 10 real + 10 fake / ImageNet class;
- main: khóa sau R0 audit + compute profiling.

Sampling phải deterministic và stratified theo class khi metadata cho phép.

### Bias policy

Không coi GenImage raw là bias-free.

R0 phải đo tối thiểu:

- file format;
- JPEG/compression quality;
- image size/resolution;
- class/source balance;
- duplicate/near-duplicate risk.

Không âm thầm normalize/re-encode để loại bias. Nếu áp dụng bias-control preprocessing, tạo decision/version riêng và rerun baseline liên quan.

## Why

Strategy này giữ experiment đầu tiên đơn giản: một training generator, một near-OOD generator, nhiều unseen generators, sau đó mới mở ra benchmark hiện đại và real-world. Nó phù hợp mục tiêu học tập + paper-oriented research mà không biến R0/R2 thành data-engineering project quá lớn.

## Sources

- GenImage official repository: https://github.com/GenImage-Dataset/GenImage
- GenImage++ dataset: https://huggingface.co/datasets/Lunahera/genimagepp
- Unbiased GenImage: https://github.com/gendetection/UnbiasedGenImage
- WildRF repository: https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld
- Chameleon/AIDE: https://github.com/shilinyan99/AIDE

## Consequence

- R0 inventory/split/audit code phải hỗ trợ generator metadata và deterministic scale manifests.
- R2 ablation semantic-only / forensic-only / fusion phải dùng cùng sealed manifests.
- External benchmark data không được dùng để tune Stage 1 hyperparameter hoặc threshold.
- Nếu R0 phát hiện bias nghiêm trọng không thể kiểm soát hợp lý, dataset strategy phải được review lại trước R2.
