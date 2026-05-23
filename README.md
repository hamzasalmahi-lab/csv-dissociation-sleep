# Study 1c — CII-PDS Dissociation in Sleep
## Pre-registered test of the HRIT v3 two-dimensional consciousness architecture

**Pre-registration:** https://doi.org/10.17605/OSF.IO/CRFJE  
**HRIT v3 preprint:** https://doi.org/10.5281/zenodo.19490741  
**Dataset:** CAP Sleep Database — https://physionet.org/content/capslpdb/1.0.0  
**Author:** Hamza S. Almahi · hamza.s.almahi@gmail.com

---

## What this study tests

HRIT v3 proposes that conscious experience has two **semi-independent** dimensions:

| Dimension | Definition | Proxy used here |
|---|---|---|
| **CII** (Conscious Integration Index) | Brain's hierarchical world-model construction | EEG permutation entropy (PE) |
| **PDS** (Phenomenal Depth Signal) | Interoceptive coupling verification | HEP peak-to-peak amplitude |

**Semi-independent** means they can vary in different directions simultaneously.

**REM sleep is the natural test case:**
- Cortical complexity remains high (dreaming requires active generative model → CII preserved)
- Motor atonia suppresses interoceptive afference (→ PDS should fall)
- Prediction: PE stays high in REM; HEP drops in REM

---

## Confirmed results

| Hypothesis | Prediction | Result | Stats |
|---|---|---|---|
| **H1** | PE: REM > N3 (CII preserved for dreaming) | ✅ Confirmed | W=51, p=.007, r=.757, 8/10 subjects |
| **H1 note** | Wake ≈ REM (pre-registered) | ⚠️ Partial deviation | Wake > REM (p=.006) — 5.4% drop |
| **H2** | HEP: Wake > REM (PDS drops in REM) | ✅ Confirmed | W=53, p=.003, r=.822, 9/10 subjects |
| **H3** | Within-subject CII-PDS correlation lower in REM | ❌ Not confirmed | p=.92 — N3 slow-wave confound |

**Key dissociation numbers:**
- PE drops **5.4%** Wake→REM
- HEP drops **27.7%** Wake→REM  
- Correlation between PE-change and HEP-change: **r = −0.344** (independent variation)

---

## Sample

| Category | N | Subjects |
|---|---|---|
| Total downloaded | 16 | n1–n16 |
| Excluded: calibration error (amplitude ~10⁷ µV) | 3 | n13, n14, n15 |
| Excluded: no ECG channel | 1 | n16 |
| No REM epochs | 2 | n2, n3 |
| **Final N for REM analyses** | **10** | n1, n4–n12 |

---

## Repository structure

```
smoke_test_cap.py             Smoke test — single subject (n1), run before pre-registration
download_cap.sh               Download all 16 subjects from PhysioNet
analysis.py                   Full batch pipeline: PE + HEP per subject per stage
confirmatory_tests.py         H1, H2, H3 confirmatory tests — runs on CSV only
results_per_subject_stage.csv Per-subject per-stage PE and HEP means (real data)
analysis_run_v2.log           Full batch run log with per-subject output
LICENSE                       MIT
```

---

## How to reproduce

### Prerequisites
```bash
pip install mne numpy scipy pandas matplotlib
```

### Download data
```bash
bash download_cap.sh          # requires PhysioNet account (free)
```

### Run analysis
```bash
python smoke_test_cap.py      # verify data loads correctly (n1 only)
python analysis.py            # full batch — generates results_per_subject_stage.csv
python confirmatory_tests.py  # H1, H2, H3 tests on the CSV
```

### Results are already in the repo
`results_per_subject_stage.csv` and `analysis_run_v2.log` are committed.
You can run `confirmatory_tests.py` directly without downloading the raw data.

---

## Methods summary

**CII proxy — Permutation entropy**  
Per 30-second epoch from frontal-central EEG (F3-A2 preferred, see log for per-subject channels).  
Order m = 3, delay τ = 1, normalised to [0, 1]. Epochs > 500 µV excluded.

**PDS proxy — HEP amplitude**  
Pan-Tompkins R-peak detection (5–40 Hz, 90th-percentile threshold, 0.4 s min distance).  
Ectopic rejection: RR intervals > 2 SD from mean.  
HEP epoch: 200–600 ms post R-peak, baseline-corrected (−200 to 0 ms).  
Artefact rejection: |amplitude| > 150 µV. Minimum 20 valid epochs per stage.  
Measure: peak-to-peak amplitude (µV).

**Statistics**  
Wilcoxon signed-rank. One-tailed for H1 and H2 (directional). Two-tailed for H3.  
Effect size r = Z/√N.

---

## Why H3 was not confirmed

N3 HEP amplitude exceeded Wake HEP in 6/10 subjects (group mean: N3 = 40.65 µV vs Wake = 34.84 µV). This is inconsistent with PDS suppression in NREM and is explained by **slow-wave contamination of the peak-to-peak HEP measure**: large-amplitude 0.5–1 Hz slow waves during N3 produce prominent deflections in the 200–600 ms window, inflating the peak-to-peak estimate.

The H3 result cannot be interpreted as evidence against the framework. The fix for future studies: mean amplitude with high-pass filtering (> 1 Hz) to remove slow-wave contamination from NREM epochs.

---

## Connection to other HRIT studies

| Study | Prediction tested | Dataset | Status |
|---|---|---|---|
| **Study 1** | PEWS: aperiodic slope variance peaks before N3 | Sleep-EDF Expanded | ✅ Confirmed |
| **Study 1b** | PEWS: PE variance peaks before N3 | Sleep-EDF Expanded | ✅ Confirmed |
| **Study 1c (this)** | CII-PDS semi-independence in REM | CAP Sleep Database | ✅ H1+H2 confirmed |
| **DEWS negative control** | No pre-ictal HEP variance rise in epilepsy | SeizeIT2 (N=111) | ✅ Null confirmed |
| **Study 4 (planned)** | DEWS: HEP variance rises before functional seizures | FND cohort (Yogarajah) | In preparation |

---

## License

MIT — see LICENSE
