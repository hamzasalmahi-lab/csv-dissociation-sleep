# Two-component HEP extension — Gautier 2025 framework

**Status:** Exploratory analysis. Pipeline validated on synthetic data; not yet run on real CAP recordings.

## What this tests

Building on Gautier et al. (2025, *Psychophysiology*) "Characterizing the HEP: A Two-Component Model of Cardiac Signal Processing." Gautier propose two HEP components:

| Component | Window | Site | Putative function |
|---|---|---|---|
| Early | 100–250 ms | Fronto-central (negative) | Primary cardiac signal integration; theta phase-reset mediated |
| Late | 250–500 ms | Centro-posterior (positive) | Elaborative processing |

Gautier found the late component was **not modulated by task** across emotion and tactile paradigms in 104 participants — a robust null.

**The reframe being tested here:** maybe task isn't the right modulator; **internal precision state** is. In active inference terms, the coupling between early (≈ observation channel) and late (≈ posterior / belief update) carries information about effective precision. Predictions for CAP sleep stages:

| Stage | B-precision (Whyte 2026) | Predicted early-late coupling |
|---|---|---|
| N3 | High (LC tonic high, stable belief) | High |
| Wake | Intermediate | Intermediate |
| REM | Low (LC silent, fast belief switching) | Low |

## Pipeline

```
extract_two_component.py     → results_two_component.csv  (per-subject per-stage coupling)
test_two_component.py        → Wilcoxon tests on coupling differences
synthetic_smoke_test.py      → mechanical verification on fake data (per-stage)
end_to_end_smoke.py          → full pipeline test with 10 synthetic subjects + tests
```

## Hypotheses (pre-specified)

| ID | Test | Direction | Statistic |
|---|---|---|---|
| H_coupling-1 | Coupling(N3) > Coupling(R) | One-tailed | Wilcoxon signed-rank |
| H_coupling-2 | Coupling(W) > Coupling(R) | One-tailed | Wilcoxon signed-rank |
| H_coupling-3 | Coupling(N3) vs Coupling(W) | Two-tailed (no directional prior) | Wilcoxon signed-rank |

Effect size: r = |Z| / √N. Secondary: Hedges' g paired.

## What's been verified

**Synthetic smoke test:** ✅ PASS

```
[  W]  target r=+0.50  recovered r=+0.503  PASS
[ N3]  target r=+0.80  recovered r=+0.781  PASS
[  R]  target r=+0.10  recovered r=+0.076  PASS
```

**End-to-end with 10 synthetic subjects under hypothesised structure:**

| Stage | recovered mean (synth) | target mean |
|---|---|---|
| W  | 0.445 | 0.45 |
| N3 | 0.689 | 0.75 |
| R  | 0.119 | 0.15 |

H1 (N3 > R): p = 0.0010, g = 2.26
H2 (W > R):  p = 0.0029, g = 1.22
H3 (N3 vs W, two-tailed): p = 0.0039, g = 1.31 (N3 > W)

The pipeline detects the hypothesised pattern when it exists. **It does not test whether the pattern actually exists in CAP data.**

## How to run on real data

1. Download CAP data: `bash download_cap.sh` (requires PhysioNet account)
2. Extract: `python extract_two_component.py`  → produces `results_two_component.csv`
3. Test: `python test_two_component.py`

## Key methodological caveats

- **Channel availability.** CAP's sleep montage is limited (mostly anterior derivations). The pipeline looks for a frontal-central channel for the early component AND a centro-posterior channel for the late component. If posterior isn't available, it falls back to within-channel coupling, which loses Gautier's spatial dissociation. The `cross_channel` column in the output flags which subjects used which mode.
- **Mean vs peak-to-peak.** Per Gautier, mean amplitude in the window — not peak-to-peak as in the original Study 1c. Robust for short windows.
- **Baseline:** −150 to −50 ms (Gautier convention), tighter than Study 1c's −200 to 0.
- **Minimum epochs per stage** raised from 20 to 30 — coupling correlation needs more samples for a stable estimate than a stage mean does.
- **Confound:** N3 slow waves (0.5–1 Hz) contaminated peak-to-peak HEP in the original Study 1c. Mean amplitude is less affected, but high-pass filtering above 1 Hz should still be considered as a sensitivity analysis.

## Status of the underlying hypothesis

Speculative. Three concerns to flag explicitly:

1. **No published data on early-late HEP decoupling in clinical populations.** Gautier's data are healthy participants under task manipulation. The proposed precision-state interpretation has no empirical precedent.
2. **The early-late coupling within a single recording is not a true dual-observable.** Both components are responses to the same heartbeat; they cannot provide independent timescale information the way HEP + cortical complexity would. Identifiability gain over single-observable analysis is likely modest.
3. **Gautier's empirical observation was a null result.** Reinterpreting it as supporting a different modulator is defensible but speculative; the data do not refute the precision interpretation, but they do not support it either.

If the analysis here produces a clean N3 > W > R pattern in coupling, it constitutes preliminary support for the framework — sufficient to motivate a Paper 2 (sleep) section, not sufficient to claim the mechanism. If the analysis produces null or mixed results, the framework is not refuted (sample size, channel limitations) but loses its empirical motivation.

## Connection to FND active inference paper

Not for the FND Paper 1. The single-observable Paper 1 stays as-is; this two-component framework, if it survives sleep validation, is a candidate methodological asset for **Paper 2 (sleep dual-observable analysis)** as a *within-modality* dual-observable that doesn't require a second recording. Paper 3 (dual-observable FND) would then test the same coupling structure in pre-ictal data.
