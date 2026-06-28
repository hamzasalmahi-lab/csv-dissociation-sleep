"""
Synthetic smoke test for the two-component HEP pipeline.

Generates fake EEG + ECG + sleep staging with known early-late coupling
that varies by stage, then runs extract_two_component_hep to verify the
pipeline recovers the ground-truth coupling structure.

Ground truth:
    Stage W   → r = 0.50  (moderate coupling)
    Stage N3  → r = 0.80  (high coupling)
    Stage R   → r = 0.10  (low coupling)

Passes if recovered r is within ±0.15 of ground truth per stage.
"""

import numpy as np
import pandas as pd
from extract_two_component import (
    detect_rpeaks, extract_two_component_hep, stage_coupling, MIN_EPOCHS,
    BASE_START_MS, BASE_END_MS, EARLY_START_MS, EARLY_END_MS,
    LATE_START_MS, LATE_END_MS,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SFREQ = 512
DURATION_SEC_PER_STAGE = 1200  # 20 min per stage → plenty of R-peaks for coupling
STAGES_GT = [
    ('W',  0.50),
    ('N3', 0.80),
    ('R',  0.10),
]
RNG = np.random.default_rng(42)


def synth_ecg(n_samples, sfreq, hr_bpm=65):
    """Synthetic ECG with regular R-peaks at hr_bpm + small jitter."""
    rr_sec = 60.0 / hr_bpm
    rr_samples = int(rr_sec * sfreq)
    ecg = np.zeros(n_samples)
    # Sharp R-peak deflection every rr_samples (plus a few samples of jitter)
    peak_times = []
    t = rr_samples
    while t < n_samples - 10:
        jitter = RNG.integers(-int(0.05 * sfreq), int(0.05 * sfreq))
        peak_pos = t + jitter
        if 0 <= peak_pos < n_samples - 5:
            # Stylized QRS: -0.1, +1.0, -0.3
            ecg[peak_pos - 1] = -0.1e-3
            ecg[peak_pos]     = +1.0e-3
            ecg[peak_pos + 1] = -0.3e-3
            peak_times.append(peak_pos)
        t += rr_samples
    ecg += RNG.normal(0, 5e-5, n_samples)  # baseline noise
    return ecg, np.array(peak_times)


def synth_eeg_with_hep(n_samples, sfreq, rpeak_samples,
                       early_dev_per_trial, late_dev_per_trial,
                       early_amp_uv, late_amp_uv):
    """
    Build EEG with HEP-shaped responses time-locked to R-peaks.
    Pre-computed per-trial deviations passed in so cross-channel correlation
    can be controlled at the caller.
    """
    eeg = RNG.normal(0, 5e-6, n_samples)  # 5 µV background noise

    early_s = int(EARLY_START_MS / 1000 * sfreq)
    early_e = int(EARLY_END_MS   / 1000 * sfreq)
    late_s  = int(LATE_START_MS  / 1000 * sfreq)
    late_e  = int(LATE_END_MS    / 1000 * sfreq)

    for i, p in enumerate(rpeak_samples):
        if p + late_e >= n_samples or i >= len(early_dev_per_trial):
            continue
        e_amp = (early_amp_uv + early_dev_per_trial[i]) * 1e-6
        l_amp = (late_amp_uv  + late_dev_per_trial[i])  * 1e-6
        eeg[p + early_s:p + early_e] += e_amp
        eeg[p + late_s:p + late_e]   += l_amp
    return eeg


def main():
    print("=" * 64)
    print("SYNTHETIC SMOKE TEST — two-component HEP pipeline")
    print("=" * 64)
    print(f"sfreq           = {SFREQ} Hz")
    print(f"per-stage dur   = {DURATION_SEC_PER_STAGE} s")
    print(f"min epochs/stage= {MIN_EPOCHS}")
    print(f"target stages   = {[s for s, _ in STAGES_GT]}")
    print()

    rows = []
    pass_all = True
    for stage_name, target_r in STAGES_GT:
        n = DURATION_SEC_PER_STAGE * SFREQ
        ecg, rpeak_samples = synth_ecg(n, SFREQ)

        # Inject ground-truth amplitudes
        early_mu = {'W':  3.0, 'N3':  5.0, 'R':  2.0}[stage_name]
        late_mu  = {'W': -2.0, 'N3': -4.0, 'R': -1.0}[stage_name]

        # Pre-compute correlated trial-level deviations ONCE per stage —
        # so they're identical across the two channels
        sigma = 8.0
        n_trials = len(rpeak_samples)
        z1 = RNG.standard_normal(n_trials)
        z2 = RNG.standard_normal(n_trials)
        e_dev = sigma * z1
        l_dev = sigma * (target_r * z1 + np.sqrt(1 - target_r**2) * z2)

        eeg_front = synth_eeg_with_hep(n, SFREQ, rpeak_samples,
                                        e_dev, l_dev, early_mu, late_mu)
        # Post channel: same per-trial deviations, slightly different mean
        # (different scalp location but same underlying neural process)
        eeg_post  = synth_eeg_with_hep(n, SFREQ, rpeak_samples,
                                        e_dev, l_dev,
                                        early_mu * 0.6, late_mu * 1.3)

        # Use INJECTED peak positions to isolate extraction logic
        early, late, _ = extract_two_component_hep(
            eeg_front, eeg_post, rpeak_samples, SFREQ)
        mean_e, mean_l, r, slope, n_eps = stage_coupling(early, late)

        ok_r = (not np.isnan(r)) and abs(r - target_r) < 0.15
        verdict = "PASS" if ok_r else "FAIL"
        if not ok_r:
            pass_all = False

        print(f"[{stage_name:>3}]  target r={target_r:+.2f}  "
              f"recovered r={r:+.3f}  "
              f"n_epochs={n_eps:4d}  "
              f"early={mean_e:+5.2f}µV  late={mean_l:+5.2f}µV   {verdict}")
        rows.append({'stage': stage_name, 'target_r': target_r,
                     'recovered_r': r, 'n': n_eps, 'ok': ok_r})

    print()
    print("=" * 64)
    print(f"OVERALL: {'PASS' if pass_all else 'FAIL'}")
    print("=" * 64)

    pd.DataFrame(rows).to_csv("smoke_test_two_component.csv", index=False)
    print("\nSaved: smoke_test_two_component.csv")
    return 0 if pass_all else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
