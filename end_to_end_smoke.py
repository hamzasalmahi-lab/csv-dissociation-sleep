"""
End-to-end pipeline test using a synthetic cohort.

Generates 10 fake subjects with stage-dependent early-late coupling that
matches the hypothesised direction (N3 > W > R), then runs the full
confirmatory test pipeline to verify the analysis chain works correctly.

This is a SMOKE TEST OF THE ANALYSIS, not evidence about the hypothesis.
Real evidence requires running extract_two_component.py on actual CAP data.
"""

import numpy as np
import pandas as pd
from synthetic_smoke_test import synth_ecg, synth_eeg_with_hep
from extract_two_component import (
    detect_rpeaks, extract_two_component_hep, stage_coupling, MIN_EPOCHS,
)

SFREQ = 512
N_SUBJECTS = 10
DURATION_PER_STAGE = 1200  # 20 min/stage

# Hypothesised direction with between-subject variability
# (Subject-specific coupling perturbed around stage-typical values)
STAGE_R_MEAN = {'W': 0.45, 'N3': 0.75, 'R': 0.15}
STAGE_R_SUBJECT_SIGMA = 0.15  # between-subject variability


def generate_subject(sub_id, rng):
    """Generate one fake subject's per-stage results."""
    rows = []
    for stage in ['W', 'N3', 'R']:
        # Subject-specific true coupling for this stage
        target_r = np.clip(
            STAGE_R_MEAN[stage] + rng.normal(0, STAGE_R_SUBJECT_SIGMA),
            -0.95, 0.95)

        n_samples = DURATION_PER_STAGE * SFREQ
        ecg, peaks = synth_ecg(n_samples, SFREQ)
        n_trials = len(peaks)

        # Trial-level (e_dev, l_dev) with target correlation
        sigma = 8.0
        z1 = rng.standard_normal(n_trials)
        z2 = rng.standard_normal(n_trials)
        e_dev = sigma * z1
        l_dev = sigma * (target_r * z1 + np.sqrt(1 - target_r**2) * z2)

        early_mu = {'W': 3.0, 'N3': 5.0, 'R': 2.0}[stage]
        late_mu  = {'W': -2.0, 'N3': -4.0, 'R': -1.0}[stage]

        eeg_front = synth_eeg_with_hep(
            n_samples, SFREQ, peaks, e_dev, l_dev, early_mu, late_mu)
        eeg_post = synth_eeg_with_hep(
            n_samples, SFREQ, peaks, e_dev, l_dev,
            early_mu * 0.6, late_mu * 1.3)

        early, late, _ = extract_two_component_hep(
            eeg_front, eeg_post, peaks, SFREQ)
        mean_e, mean_l, r, slope, n = stage_coupling(early, late)

        rows.append({
            'sub':            sub_id,
            'stage':          stage,
            'early_mean':     mean_e,
            'late_mean':      mean_l,
            'coupling_r':     r,
            'coupling_slope': slope,
            'n_epochs':       n,
            'front_ch':       'SYNTH_F',
            'post_ch':        'SYNTH_P',
            'cross_channel':  True,
            'target_r':       target_r,
        })
    return rows


def main():
    print("=" * 64)
    print(f"END-TO-END SMOKE TEST — {N_SUBJECTS} synthetic subjects")
    print("=" * 64)
    print(f"Stage-typical coupling: {STAGE_R_MEAN}")
    print(f"Between-subject σ:      {STAGE_R_SUBJECT_SIGMA}")
    print()

    rng = np.random.default_rng(seed=20260628)
    all_rows = []
    for i in range(1, N_SUBJECTS + 1):
        sub_id = f"synth_{i:02d}"
        print(f"  {sub_id} ...", end=" ", flush=True)
        try:
            rows = generate_subject(sub_id, rng)
            all_rows.extend(rows)
            r_per_stage = {row['stage']: row['coupling_r'] for row in rows}
            print(f"r(W,N3,R) = ({r_per_stage['W']:+.2f}, "
                  f"{r_per_stage['N3']:+.2f}, {r_per_stage['R']:+.2f})")
        except Exception as e:
            print(f"FAIL — {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv("results_two_component.csv", index=False)
    print(f"\nWrote results_two_component.csv  ({len(df)} rows)")
    print()
    print("=" * 64)
    print("RUNNING CONFIRMATORY TESTS")
    print("=" * 64)

    import subprocess
    result = subprocess.run(
        ["python3", "test_two_component.py"],
        capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)


if __name__ == "__main__":
    main()
