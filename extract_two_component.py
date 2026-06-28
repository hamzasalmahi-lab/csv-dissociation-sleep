"""
Two-component HEP extraction — Gautier et al. 2025 framework applied to CAP sleep data.

Modifies analysis.py to extract early (100-250 ms, fronto-central) and late
(250-500 ms, centro-posterior) HEP components separately, then computes the
per-stage coupling between them as the candidate precision-state marker.

Hypothesis (active inference framing):
    Early component  ≈ likelihood / observation channel (bottom-up cardiac integration)
    Late component   ≈ posterior / belief-updating channel (top-down elaboration)
    Coupling (r)     ≈ how tightly the brain integrates observation into belief
                       — i.e., a candidate read-out of effective precision

Predictions for CAP sleep stages:
    N3   (high B-precision per Whyte 2026)     → high early-late coupling
    REM  (low B-precision)                     → low early-late coupling
    Wake (intermediate)                        → intermediate coupling

Outputs per-subject per-stage CSV with:
    early_mean, late_mean       — mean component amplitudes (µV)
    coupling_r                  — Pearson correlation of early vs late across epochs
    coupling_slope              — linear regression slope (early → late)
    n_epochs                    — number of valid epochs contributing
    front_ch, post_ch           — channels used (post_ch = NaN if cross-channel unavailable)
    cross_channel               — True if early and late used different channels

Run after download_cap.sh.
"""

import mne
import numpy as np
import pandas as pd
import re
import math
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import pearsonr, linregress

# ── CONFIG ────────────────────────────────────────────────────────────────────
SFREQ_TARGET    = 512
EARLY_START_MS  = 100      # early component window
EARLY_END_MS    = 250
LATE_START_MS   = 250      # late component window
LATE_END_MS     = 500
BASE_START_MS   = -150     # tighter baseline per Gautier (-150 to -50 ms)
BASE_END_MS     = -50
REJECT_UV       = 150
RR_SD_THRESH    = 2
MIN_EPOCHS      = 30       # raised from 20 — coupling needs enough samples for stable r

# Frontal-central channel priority (early component site)
FRONT_PRIORITY = ['F3-A2', 'F4-A2', 'F3-A1', 'F4-A1',
                  'FC1', 'FC2', 'FCZ', 'F3', 'F4',
                  'C3-A2', 'C4-A2', 'C3', 'C4', 'CZ']

# Centro-posterior channel priority (late component site)
POST_PRIORITY = ['CP3', 'CP4', 'CPZ', 'P3-A2', 'P4-A2', 'P3-A1', 'P4-A1',
                 'P3', 'P4', 'PZ', 'O1-A2', 'O2-A2', 'O1-A1', 'O2-A1',
                 'O1', 'O2', 'OZ']

STAGE_MAP = {
    'SLEEP-S0':  'W',
    'SLEEP-S1':  'N1',
    'SLEEP-S2':  'N2',
    'SLEEP-S3':  'N3',
    'SLEEP-S4':  'N3',
    'SLEEP-REM': 'R',
}


# ── STAGING PARSER (lifted from analysis.py) ─────────────────────────────────
def parse_staging(txt_path):
    time_pattern = re.compile(r'^\d{2}:\d{2}:\d{2}$')
    parsed = []
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            time_idx = next(
                (i for i, p in enumerate(parts) if time_pattern.match(p.strip())),
                None)
            if time_idx is None or time_idx + 1 >= len(parts):
                continue
            event = parts[time_idx + 1].strip()
            if event not in STAGE_MAP:
                continue
            time_str = parts[time_idx].strip()
            h, m, s = map(int, time_str.split(':'))
            t_sec = h * 3600 + m * 60 + s
            parsed.append({
                'stage':    STAGE_MAP[event],
                'time_sec': t_sec,
                'duration': 30,
            })
    df = pd.DataFrame(parsed)
    if len(df) > 0:
        t0 = df['time_sec'].iloc[0]
        df['elapsed_sec'] = df['time_sec'].apply(
            lambda t: t - t0 if t >= t0 else t + 86400 - t0)
    return df


# ── CHANNEL SELECTION ─────────────────────────────────────────────────────────
def find_channel(ch_names_upper, priority_list, exclude_idx=None):
    """Return (idx, original_name) for the first priority match, or (None, None)."""
    for target in priority_list:
        for i, c in enumerate(ch_names_upper):
            if exclude_idx is not None and i == exclude_idx:
                continue
            if target in c:
                return i, c
    return None, None


# ── R-PEAK DETECTION ──────────────────────────────────────────────────────────
def detect_rpeaks(ecg_signal, sfreq):
    """Pan-Tompkins style with ectopic rejection. Returns R-peak sample indices."""
    ecg_filt = mne.filter.filter_data(
        ecg_signal, sfreq, l_freq=5, h_freq=40, verbose=False)
    ecg_sq = ecg_filt ** 2
    peaks, _ = find_peaks(
        ecg_sq,
        distance=int(0.4 * sfreq),
        height=np.percentile(ecg_sq, 90))
    if len(peaks) < 3:
        return peaks
    rr = np.diff(peaks) / sfreq
    rr_mean, rr_std = np.mean(rr), np.std(rr)
    valid = [peaks[0]]
    for i, p in enumerate(peaks[1:], 1):
        rr_i = (p - peaks[i-1]) / sfreq
        if abs(rr_i - rr_mean) < RR_SD_THRESH * rr_std:
            valid.append(p)
    return np.array(valid)


# ── TWO-COMPONENT HEP EXTRACTION ─────────────────────────────────────────────
def extract_two_component_hep(eeg_front, eeg_post, rpeaks, sfreq):
    """
    Extract early (front) and late (post) component amplitudes per heartbeat.
    Both signals must be in volts.

    Returns: (early_uv, late_uv, hep_times_sec) — three 1D arrays of same length.
    `eeg_post` may be None — in which case late is taken from the same channel
    as early (within-channel mode).
    """
    if eeg_post is None:
        eeg_post = eeg_front

    base_s = int(BASE_START_MS  / 1000 * sfreq)
    base_e = int(BASE_END_MS    / 1000 * sfreq)
    early_s = int(EARLY_START_MS / 1000 * sfreq)
    early_e = int(EARLY_END_MS   / 1000 * sfreq)
    late_s  = int(LATE_START_MS  / 1000 * sfreq)
    late_e  = int(LATE_END_MS    / 1000 * sfreq)

    early_uv, late_uv, times = [], [], []

    for p in rpeaks:
        b_start, b_end = p + base_s, p + base_e
        e_start, e_end = p + early_s, p + early_e
        l_start, l_end = p + late_s, p + late_e

        if b_start < 0 or l_end >= len(eeg_front) or l_end >= len(eeg_post):
            continue

        # Baseline from front channel
        base_front = np.mean(eeg_front[b_start:b_end])
        base_post  = np.mean(eeg_post[b_start:b_end])

        # Early window from front channel — mean amplitude
        early_seg = eeg_front[e_start:e_end] - base_front
        late_seg  = eeg_post[l_start:l_end]  - base_post

        # Artifact rejection on either window
        max_uv = max(np.max(np.abs(early_seg)),
                     np.max(np.abs(late_seg))) * 1e6
        if max_uv > REJECT_UV:
            continue

        # Mean amplitude per window (µV)
        # Mean (not peak-to-peak) per Gautier — more robust for short windows
        early_uv.append(np.mean(early_seg) * 1e6)
        late_uv.append(np.mean(late_seg)  * 1e6)
        times.append(p / sfreq)

    return np.array(early_uv), np.array(late_uv), np.array(times)


# ── PER-STAGE COUPLING ────────────────────────────────────────────────────────
def stage_coupling(early, late):
    """Return (mean_early, mean_late, pearson_r, slope, n)."""
    n = len(early)
    if n < MIN_EPOCHS:
        return np.nan, np.nan, np.nan, np.nan, n
    if np.std(early) < 1e-6 or np.std(late) < 1e-6:
        return np.mean(early), np.mean(late), np.nan, np.nan, n
    r, _ = pearsonr(early, late)
    slope = linregress(early, late).slope
    return np.mean(early), np.mean(late), r, slope, n


# ── SUBJECT PROCESSOR ─────────────────────────────────────────────────────────
def process_subject(sub_id, data_dir="cap_data"):
    edf_path = Path(data_dir) / f"{sub_id}.edf"
    txt_path = Path(data_dir) / f"{sub_id}.txt"

    if not edf_path.exists() or not txt_path.exists():
        print(f"[SKIP] {sub_id}: files not found")
        return None

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    except Exception as e:
        print(f"[SKIP] {sub_id}: EDF load — {e}")
        return None

    sfreq = raw.info['sfreq']
    ch_upper = [c.upper() for c in raw.ch_names]

    # ECG
    ecg_idx = next((i for i, c in enumerate(ch_upper)
                    if any(k in c for k in ['EKG', 'ECG', 'CARD'])), None)
    if ecg_idx is None:
        print(f"[SKIP] {sub_id}: no ECG")
        return None

    # Front (early) and post (late) channels
    front_idx, front_name = find_channel(ch_upper, FRONT_PRIORITY,
                                          exclude_idx=ecg_idx)
    post_idx,  post_name  = find_channel(ch_upper, POST_PRIORITY,
                                          exclude_idx=ecg_idx)

    if front_idx is None:
        print(f"[SKIP] {sub_id}: no frontal-central EEG")
        return None

    cross_channel = post_idx is not None
    mode = "CROSS-channel" if cross_channel else "within-channel"

    ecg_signal   = raw.get_data(picks=[ecg_idx])[0]
    front_signal = raw.get_data(picks=[front_idx])[0]
    post_signal  = raw.get_data(picks=[post_idx])[0] if cross_channel else None

    print(f"[{sub_id}] ECG: {raw.ch_names[ecg_idx]} | "
          f"FRONT: {front_name} | POST: {post_name if cross_channel else '(within)'} | "
          f"mode: {mode}")

    # R-peaks
    rpeaks = detect_rpeaks(ecg_signal, sfreq)
    if len(rpeaks) < 100:
        print(f"[SKIP] {sub_id}: only {len(rpeaks)} R-peaks")
        return None

    # All HEP epochs across the whole recording
    early, late, times = extract_two_component_hep(
        front_signal, post_signal, rpeaks, sfreq)
    print(f"[{sub_id}] HEP epochs: {len(early)} valid")

    # Staging
    df_stages = parse_staging(txt_path)
    if len(df_stages) == 0:
        print(f"[SKIP] {sub_id}: no stages")
        return None

    # Per-stage coupling
    results = []
    for stage in ['W', 'N1', 'N2', 'N3', 'R']:
        stage_rows = df_stages[df_stages.stage == stage]
        if len(stage_rows) == 0:
            continue
        # Build mask: time in any 30-s epoch of this stage
        mask = np.zeros(len(times), dtype=bool)
        for _, row in stage_rows.iterrows():
            t0 = row['elapsed_sec']
            t1 = t0 + row['duration']
            mask |= (times >= t0) & (times < t1)
        e_stage, l_stage = early[mask], late[mask]
        mean_e, mean_l, r, slope, n = stage_coupling(e_stage, l_stage)
        results.append({
            'sub':            sub_id,
            'stage':          stage,
            'early_mean':     mean_e,
            'late_mean':      mean_l,
            'coupling_r':     r,
            'coupling_slope': slope,
            'n_epochs':       n,
            'front_ch':       front_name,
            'post_ch':        post_name if cross_channel else 'WITHIN',
            'cross_channel':  cross_channel,
        })
        print(f"  {stage}: n={n:5d}  early={mean_e:+6.2f}µV  "
              f"late={mean_l:+6.2f}µV  r={r:+.3f}  slope={slope:+.3f}")

    return pd.DataFrame(results)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    subjects = [f"n{i}" for i in range(1, 17)]
    all_results = []
    for sub in subjects:
        print(f"\n{'='*60}")
        result = process_subject(sub)
        if result is not None:
            all_results.append(result)

    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        df.to_csv("results_two_component.csv", index=False)
        print(f"\n{'='*60}")
        print(f"Saved: results_two_component.csv")
        print(f"Subjects: {df['sub'].nunique()}  |  "
              f"Cross-channel subjects: {df['cross_channel'].any() and df.groupby('sub')['cross_channel'].first().sum()}")
        print("\nPer-stage coupling means (Pearson r):")
        print(df.groupby('stage')['coupling_r'].agg(['mean', 'std', 'count']).round(3))
