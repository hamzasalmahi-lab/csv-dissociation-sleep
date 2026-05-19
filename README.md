# CII-PDS Dissociation Study

**Pre-registration:** https://doi.org/10.17605/OSF.IO/CRFJE  
**Framework:** HRIT v3 — https://doi.org/10.5281/zenodo.19490741  
**Dataset:** CAP Sleep Database, PhysioNet — https://physionet.org/content/capslpdb/1.0.0/

---

## What this study tests

HRIT v3 proposes that conscious experience requires two semi-independent computations:

- **CII (Presence)** — the brain's construction of a world-model. Proxy: EEG complexity (permutation entropy).
- **PDS (Depth)** — the brain's ongoing verification of biological existence. Proxy: HEP amplitude (heartbeat-evoked potential).

Semi-independent means either can vary independently of the other. This study tests whether they dissociate during **REM sleep** — where cortical activity is high (CII should be high) but interoceptive afference is suppressed by atonia (PDS should be low).

---

## Pre-registered hypotheses

| Hypothesis | Prediction |
|---|---|
| H1 | Permutation entropy: Wake > REM ≈ N1 > N2 > N3 |
| H2 | HEP amplitude: Wake > REM, despite REM PE remaining high |
| H3 | Within-subject CII-PDS correlation lower in REM than NREM |

---

## Results

| Hypothesis | Result | Stats |
|---|---|---|
| H1 (PE REM > N3) | **Confirmed** | p = .0068, r = 0.757, 8/10 subjects |
| H1 note (Wake ≈ REM) | Partial deviation | Wake > REM, p = .0059, r = 0.822 |
| H2 (Wake HEP > REM HEP) | **Confirmed** | p = .0029, r = 0.822, 9/10 subjects |
| H3 (correlation dissociation) | Not confirmed | p = .92 — N3 slow-wave confound |

**Key finding:** HEP amplitude drops 28% from wake to REM. Permutation entropy drops only 10%. Correlation between the two changes across subjects: r = −0.344. The two dimensions changed independently — confirming semi-independence for the wake/REM contrast.

**H3 note:** Peak-to-peak HEP is contaminated by slow-wave activity in N3 (N3 HEP > Wake HEP in 6/10 subjects). H3 cannot be interpreted as evidence against the framework — it is a measurement limitation of peak-to-peak amplitude during NREM. Mean amplitude with high-pass filtering is recommended for future NREM studies.

---

## Sample

- **16 healthy controls** (n1–n16), CAP Sleep Database
- **Excluded:** n13, n14, n15 (EEG calibration error — amplitude ~10⁷ µV); n16 (no ECG channel)
- **No REM data:** n2, n3 (excluded from H2 and H3)
- **Final N = 10** for primary dissociation analyses

---

## Repository structure

```
smoke_test_cap.py            — initial data verification (n1 only, pre-registration)
analysis.py                  — full pipeline: staging parser, PE, HEP extraction, batch loop
confirmatory_tests.py        — H1, H2, H3 tests (runs on CSV only, no raw data)
results_per_subject_stage.csv — per-subject per-stage PE and HEP means
analysis_run_v2.log           — full batch run log
cap_data/                    — downloaded EDF and TXT files (not tracked by git)
```

---

## Methods summary

**CII proxy:** Permutation entropy (order m=3, delay τ=1, normalised), computed per 30-second sleep epoch from frontal-central EEG channel. Epochs >500µV excluded.

**PDS proxy:** HEP peak-to-peak amplitude, 200–600ms post R-peak, baseline corrected (−200 to 0ms). R-peaks detected via Pan-Tompkins (5–40Hz bandpass). Ectopic beats excluded (RR >2 SD from mean). Epoch rejection threshold: 150µV. Minimum 20 valid epochs per stage required.

**Statistics:** Wilcoxon signed-rank, effect size r = Z/√N. One-tailed for directional predictions (H1, H2). Bonferroni correction across H1 pairwise comparisons (α = 0.0125).

---

## Connection to ongoing work

This study provides pre-registered support for the two-dimensional architecture of HRIT v3 in healthy sleep. Study 4 (in collaboration with Dr Mahinda Yogarajah, UCL Queen Square) tests the DEWS prediction — HEP amplitude variance rising before functional seizure onset — in an FND cohort. A negative control (N=115 epileptic seizures, SeizeIT2 dataset) shows no consistent pre-ictal variance rise in epilepsy: github.com/hamzasalmahi-lab/ds005873-FND

---

## Citation

Almahi, H. S. (2026). CII-PDS Dissociation Study — Testing Semi-Independence of Presence and Depth Across Sleep Stages. OSF. https://doi.org/10.17605/OSF.IO/CRFJE

Framework preprint: Almahi, H. S. (2026). HRIT v3. Zenodo. https://doi.org/10.5281/zenodo.19490741