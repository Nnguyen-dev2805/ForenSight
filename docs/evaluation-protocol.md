# Evaluation Protocol

## 1. Mục tiêu

Tạo protocol cố định để so sánh mọi experiment công bằng.

Không thay protocol sau khi nhìn kết quả nếu không ghi rõ decision và rerun baseline liên quan.

## 2. Split policy

### Train / validation / test

Mọi split phải deterministic và được lưu bằng manifest.

- **Train:** dùng để optimize model.
- **Validation:** chọn threshold, early stopping và hyperparameter nhỏ.
- **Test:** chỉ dùng để report kết quả cuối của experiment đã khóa config.

Không dùng test set để chọn threshold hoặc tune hyperparameter.

### Cross-generator

Primary cross-generator protocol là **generator-disjoint**:

```text
Train generators: G_train
Validation generators: seen distribution hoặc split đã định nghĩa trước
Test generators: G_unseen

G_unseen ∩ G_train = ∅
```

Danh sách generator train/test phải được khóa trong manifest trước khi train.

Nếu dataset và compute cho phép, có thể thêm **leave-one-generator-out** như secondary analysis. Đây không phải yêu cầu bắt buộc cho baseline đầu tiên.

### Project protocol v1

Stage 1 baseline đầu tiên khóa protocol:

- train: GenImage SD1.4;
- in-domain: held-out SD1.4;
- near-OOD: SD1.5;
- cross-generator OOD: Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN;
- modern external: GenImage++;
- real-world external: WildRF;
- optional external: Chameleon nếu access có sẵn.

External test sets không được dùng để tune hyperparameter hoặc threshold.

GenImage raw data phải qua compression/resolution bias audit trước khi score được dùng cho kết luận research. Nếu preprocessing bias-control được thay đổi, protocol version phải đổi và baseline liên quan phải rerun.

### Cross-dataset

Train trên dataset A, test trực tiếp trên dataset B mà không tune bằng test B.

Nếu cần threshold cho dataset B, dùng threshold đã chọn từ validation của training protocol hoặc một validation set độc lập được khai báo trước.

## 3. Evaluation axes

### A. In-domain

Train/test cùng dataset hoặc cùng generator distribution.

Mục đích: sanity check khả năng học task.

### B. Cross-generator

Test trên generator không xuất hiện trong train.

Đây là trục chính của RQ1.

### C. Cross-dataset

Train trên dataset A, test trên dataset B.

Mục đích: kiểm tra dataset/source shortcut.

### D. Robustness

Tối thiểu cân nhắc:

- JPEG compression;
- resize;
- crop;
- blur.

Mỗi perturbation phải có mức severity cố định và được log.

## 4. Detection metrics

### Primary metric

- **AUROC** là metric chính cho detection/generalization vì không phụ thuộc một threshold duy nhất.

### Secondary metrics

- F1-score;
- Accuracy.

Có thể thêm EER nếu cần để so sánh literature.

### Threshold policy

Threshold cho F1/Accuracy phải được chọn trên **validation set** và sau đó giữ nguyên khi chạy test tương ứng.

Không chọn threshold tối ưu trực tiếp trên test set.

## 5. Repeated runs và uncertainty

Với model có training stochastic, baseline chính nên chạy tối thiểu **3 seeds** khi compute cho phép.

Report:

```text
mean ± standard deviation
```

Nếu chỉ chạy được 1 seed vì compute, phải ghi rõ đây là preliminary result và không dùng chênh lệch nhỏ để kết luận method tốt hơn.

Không bắt buộc statistical significance test ở baseline đầu tiên.

## 6. Shortcut / leakage checks

Trước khi tin detection score, kiểm tra:

- semantic category imbalance;
- source/dataset imbalance;
- image resolution;
- file format;
- compression quality;
- obvious watermark/text;
- duplicate hoặc near-duplicate giữa splits;
- generator leakage.

Nếu có thể, báo cáo performance theo nhóm content như:

- human;
- animal;
- landscape;
- architecture;
- illustration/anime;
- text-heavy.

## 7. Stage 1 evaluation

So sánh bắt buộc:

1. semantic-only;
2. forensic-only;
3. fusion.

Không chỉ so in-domain. Kết luận về forensic/generalization phải dựa chủ yếu trên cross-generator/cross-dataset.

Một chênh lệch nhỏ giữa các model chỉ được xem là meaningful khi ổn định qua repeated runs hoặc có effect đủ rõ so với variance.

## 8. Stage 2 evaluation

### Label retention

So sánh:

- Stage 1 classifier performance;
- Stage 2A MLLM label performance;
- Stage 2B evidence-aware performance.

### Evidence retention

Khi supervision/representation cho phép, đo:

- candidate evidence localization consistency;
- candidate evidence ranking/retention;
- consistency của local evidence trước/sau projector.

Label accuracy gần Stage 1 không đủ để kết luận evidence được preserve.

## 9. Evidence causality diagnostics

Faithfulness của explanation chỉ có ý nghĩa sau khi candidate evidence đã được kiểm tra ở prediction level.

### 9.1 Image-level intervention

So sánh:

- realistic edit/inpaint ở top candidate evidence;
- matched low-evidence control;
- random same-size control.

Không dùng black-mask result làm bằng chứng duy nhất vì có thể gây OOD artifact.

Metric chính:

```text
ΔP = P(original label | original)
   - P(original label | intervention)
```

Có thể thêm deletion/insertion AUC và label flip rate như metric phụ.

### 9.2 Feature/token intervention

Ưu tiên replacement bằng representation hợp lệ/matched control hơn zero/shuffle thuần túy.

Zero/shuffle chỉ coi như stress test vì có thể tạo hidden-state OOD.

### 9.3 Necessity + sufficiency

- **Deletion:** remove evidence → confidence giảm?
- **Insertion:** restore/add evidence → confidence phục hồi?

Dùng cả hai sẽ mạnh hơn chỉ deletion.

## 10. Explanation quality

Có thể đánh giá:

- correctness;
- specificity;
- logical consistency;
- factual consistency với image;
- instruction following.

LLM-as-judge là metric bổ sung, không phải bằng chứng duy nhất của faithfulness.

## 11. Explanation faithfulness

Sau Stage 3, dùng cùng intervention để kiểm tra explanation response.

Nếu evidence bị remove/replace, explanation tương ứng phải phản ứng có hệ thống.

Ví dụ:

```text
Before: eye + hair evidence
After remove eye: explanation không nên tiếp tục khẳng định eye evidence nếu không còn quan sát hỗ trợ.
```

Faithfulness và forensic correctness là hai trục khác nhau. Một explanation có thể faithful với shortcut sai của model.

## 12. Reproducibility

Mỗi experiment phải lưu tối thiểu:

- config;
- seed;
- dataset/split version;
- threshold source/value nếu metric cần threshold;
- model/backbone version;
- checkpoint path/id;
- metrics;
- git commit nếu repo đã version control;
- timestamp;
- hardware khi ảnh hưởng kết quả.

## 13. Kết luận được phép

Không dùng các từ sau nếu chưa có evidence tương ứng:

- `generalizable` nếu chỉ test in-domain;
- `faithful` nếu chỉ có LLM/human quality score;
- `evidence-preserving` nếu chỉ đo label accuracy;
- `better` nếu baseline/protocol không tương đương hoặc chênh lệch nằm trong variance chưa được hiểu.
