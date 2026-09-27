# Decision: Lock ForenSight Milestone R2 Experimental Backbone and 81-Run Evaluation Matrix

Date: 2026-09-27  
Status: Accepted  

## Context

Following Milestone R0 dataset audit on Kaggle Tiny-GenImage (`yangsangtai/tiny-genimage`), an exploratory preprocessing pilot (Protocol A vs Protocol B1) was conducted to test whether uniform in-memory JPEG recompression at $Q=95$ could mitigate the severe format confound (Real images are 100% JPEG while Fake images are 100% PNG).

In this exploratory pilot, B1 reduced mean sensitivity ($S_{\text{B1}}=0.0162$ vs $S_{\text{A}}=0.0181$) while preserving validation AUROC within the chosen tolerance ($0.9349 \pm 0.0118$ vs $0.9403 \pm 0.0056$). However, evidence was mixed across individual seeds (on Seed 42, $S_{\text{B1}} = 0.0180$ vs $S_{\text{A}} = 0.0159$, where B1 was more sensitive than A) and does not establish compression shortcut removal. Adopting an intervention prematurely would bypass the clean, unadulterated baseline. Per `AGENTS.md`, ForenSight's canonical scientific workflow is:

$$\text{Simple Baseline} \longrightarrow \text{Observed Failure} \longrightarrow \text{Hypothesis} \longrightarrow \text{Minimal Intervention (Ablation)} \longrightarrow \text{Re-evaluate}.$$

We therefore lock the unadulterated baseline first, establish complete empirical baselines across all generators, and retain B1 strictly as an exploratory diagnostic reference and future ablation intervention if failure analysis warrants it.

## Evidence

1. **Preprocessing Pilot (B1 vs A):** Confirmed that high-frequency NPR fingerprints survive $Q=95$ recompression ($\Delta\text{AUROC} = -0.0054$, well within seed variance), while sensitivity reduction was mixed across seeds ($S_{\text{B1}} < S_{\text{A}}$ on seeds 43 and 44, but $S_{\text{B1}} > S_{\text{A}}$ on seed 42). Artifacts and logs are preserved in `reports/pilot_kaggle/` as exploratory evidence.
2. **Protocol Construction Logic Audit (`scripts/audit_all_protocols.py`):** Verified the manifest-building logic on synthetic multi-generator records (zero sample ID collisions, zero image path collisions, generator disjointness, and disjoint real allocation without replacement across Single, 7 LOGO folds, and All-7) and verified the sealed R0 real Single manifests. Production manifests for LOGO and All-7 will be audited upon generation prior to training.

## Decision

1. **Baseline Preprocessing is Locked to Canonical Isotropic Pipeline:**
   $$\text{Image (RGB)} \longrightarrow \text{Resize}(\text{short edge} = 224) \longrightarrow \text{CenterCrop}(224 \times 224) \longrightarrow \text{ToTensor()}.$$
   Zero JPEG95, zero compression normalization, zero augmentation in the primary baseline.
2. **Archival of Pilot B1:** Pilot B1 is archived as an exploratory diagnostic reference. If Milestone R2 baselines reveal anomalous OOD degradation driven by format sensitivity, B1 will be introduced as an ablation intervention.
3. **Formal Locking of the 81-Run Experimental Matrix:**
   $$\text{Total Runs} = \underbrace{1 \times 3 \times 3}_{\text{Single (9)}} + \underbrace{7 \times 3 \times 3}_{\text{LOGO (63)}} + \underbrace{1 \times 3 \times 3}_{\text{All-7 (9)}} = 81\text{ runs}.$$
   - **Architectures (3):** CLIP-only (Semantic), NPR-only (Forensic), CLIP+NPR (Fusion).
   - **Official Seeds (3):** `42`, `1337`, `2024`.
   - **Single-generator (9 runs):** Train SD1.5 $\rightarrow$ Test in-domain (SD1.5) + 6 unseen OOD generators.
   - **LOGO (63 runs across 7 folds):** Train 6 seen generators $\rightarrow$ Test 1 held-out unseen generator.
   - **All-7 (9 runs):** Train all 7 generators $\rightarrow$ Test per-generator known distributions. Closed-world upper bound; never reportable as unseen-generator generalization.

   All three protocols address **RQ1 (Forensic Perception)** — how far semantic + forensic
   representation generalizes across generators. RQ2 (Evidence Alignment) and RQ3
   (Explanation Faithfulness) are Stage-2/Stage-3 questions answered in later milestones
   (`docs/research-problem.md`); no protocol here tests them.
4. **Evaluation Invariants & Metric Suite:**
   - Decision threshold $\tau^*$ is calibrated strictly on validation data (optimal F1) and frozen for test sets. Zero test-set snooping.
   - Primary metric: **AUROC**. Secondary metrics: Accuracy, F1, Precision, Recall, TPR @ 1% FPR, TPR @ 0.1% FPR, EER, PR-AUC.
   - Reporting: $\text{mean} \pm \text{std}$ across 3 seeds with per-generator breakdowns.

## Why

This decision restores methodological clarity: it measures what the model actually learns on raw benchmarks before applying engineering interventions. Analyzing the continuum:
$$\text{Single (1 generator)} \longrightarrow \text{LOGO (6 generators)} \longrightarrow \text{All-7 (7 generators)}$$
answers the fundamental research question of how generator diversity impacts generalization and representation capacity.

## Consequence

- Execution proceeds strictly via phased gates:
  1. Protocol & manifest verification (Passed).
  2. Preprocessing baseline locking (Passed).
  3. Hyperparameter and threshold governance (Passed).
  4. Minimal smoke test across all 3 architectures (Next).
  5. Official execution of Single-generator (9 runs) on Kaggle GPU before unblocking LOGO and All-7.
