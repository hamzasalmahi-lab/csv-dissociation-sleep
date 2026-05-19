"""
CSV Dissociation Study — Analysis Pipeline
Pre-registration: https://doi.org/10.17605/OSF.IO/CRFJE

All confirmatory analyses follow the pre-registered plan exactly.
No deviation without explicit documentation.
"""

import mne
import numpy as np
import pandas as pd
import re
import math
from pathlib import Path
from scipy.signal import find_peaks
from itertools import permutations

# ── CONFIG ────────────────────────────────────────────────────────────────────
SFREQ_TARGET  = 512       # native CAP sampling rate
HEP_START_MS  = 200       # HEP epoch start post R-peak
HEP_END_MS    = 600       # HEP epoch end post R-peak
BASE_START_MS = -200      # baseline start pre R-peak
BASE_END_MS   = 0         # baseline end
REJECT_UV     = 150       # artefact rejection threshold µV
PE_ORDER      = 3         # permutation entropy order m
PE_DELAY      = 1         # permutation entropy delay τ
RR_SD_THRESH  = 2         # ectopic beat rejection: SD from mean RR
MIN_HEP_EPOCHS = 20       # minimum valid HEP epochs per stage

STAGE_MAP = {
    'SLEEP-S0':  'W',
    'SLEEP-S1':  'N1',
    'SLEEP-S2':  'N2',
    'SLEEP-S3':  'N3',
    'SLEEP-S4':  'N3',
    'SLEEP-REM': 'R',
}

# ── FUNCTION 1: PARSE STAGING ────────────────────────────────────────────────
def parse_staging(txt_path):
    """
    Parse CAP Sleep Database .txt annotation file.
    Returns DataFrame with columns: stage, time_sec, duration
    Only includes SLEEP-* events (30-second epochs).
    Excludes MCAP-A* CAP phase events.
    """
    time_pattern = re.compile(r'^\d{2}:\d{2}:\d{2}$')
    parsed = []

    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            # Find time column robustly — works for both 5 and 6 column formats
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
    # Handle midnight rollover — convert to elapsed seconds from first epoch
    if len(df) > 0:
        t0 = df['time_sec'].iloc[0]
        df['elapsed_sec'] = df['time_sec'].apply(
            lambda t: t - t0 if t >= t0 else t + 86400 - t0)
    return df


# ── FUNCTION 2: PERMUTATION ENTROPY ─────────────────────────────────────────
def permutation_entropy(signal, order=3, delay=1, normalize=True):
    """
    Compute permutation entropy (Bandt & Pompe 2002).
    signal: 1D numpy array
    order: embedding dimension m
    delay: time delay τ
    normalize: if True, divide by log(m!) to bound output to [0, 1]
    """
    n = len(signal)
    # Build all possible permutation patterns
    perms = list(permutations(range(order)))
    perm_idx = {p: i for i, p in enumerate(perms)}
    counts = np.zeros(len(perms))

    for i in range(n - (order - 1) * delay):
        pattern = tuple(
            np.argsort(signal[i:i + order * delay:delay]))
        counts[perm_idx[pattern]] += 1

    counts = counts[counts > 0]
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log(probs))

    if normalize:
        pe /= np.log(math.factorial(order))
    return pe


# ── FUNCTION 3: HEP EXTRACTION ───────────────────────────────────────────────
def extract_hep(ecg_signal, eeg_signal, sfreq,
                hep_start_ms=HEP_START_MS, hep_end_ms=HEP_END_MS,
                base_start_ms=BASE_START_MS, base_end_ms=BASE_END_MS,
                reject_uv=REJECT_UV, rr_sd_thresh=RR_SD_THRESH):
    """
    Extract heartbeat-evoked potential epochs from EEG time-locked to ECG R-peaks.

    Returns:
        hep_times: array of R-peak times in seconds
        hep_pp:    array of peak-to-peak HEP amplitude per epoch (µV)
        n_rejected: number of epochs rejected
    """
    # R-peak detection: Pan-Tompkins style
    ecg_filt = mne.filter.filter_data(
        ecg_signal, sfreq, l_freq=5, h_freq=40, verbose=False)
    ecg_sq = ecg_filt ** 2
    peaks, _ = find_peaks(ecg_sq,
        distance=int(0.4 * sfreq),
        height=np.percentile(ecg_sq, 90))

    # Ectopic beat rejection
    rr = np.diff(peaks) / sfreq
    rr_mean, rr_std = np.mean(rr), np.std(rr)
    valid_peaks = [peaks[0]]
    for i, p in enumerate(peaks[1:], 1):
        rr_i = (p - peaks[i-1]) / sfreq
        if abs(rr_i - rr_mean) < rr_sd_thresh * rr_std:
            valid_peaks.append(p)
    peaks = np.array(valid_peaks)

    # Convert ms to samples
    hep_s  = int(hep_start_ms  / 1000 * sfreq)
    hep_e  = int(hep_end_ms    / 1000 * sfreq)
    base_s = int(base_start_ms / 1000 * sfreq)  # negative
    base_e = int(base_end_ms   / 1000 * sfreq)  # 0

    hep_times  = []
    hep_pp     = []
    n_rejected = 0

    for p in peaks:
        b_start = p + base_s
        b_end   = p + base_e
        e_start = p + hep_s
        e_end   = p + hep_e

        if b_start < 0 or e_end >= len(eeg_signal):
            n_rejected += 1
            continue

        baseline = eeg_signal[b_start:b_end]
        epoch    = eeg_signal[e_start:e_end]

        if (np.max(np.abs(baseline)) * 1e6 > reject_uv or
                np.max(np.abs(epoch)) * 1e6 > reject_uv):
            n_rejected += 1
            continue

        epoch_bc = epoch - np.mean(baseline)
        pp = (np.max(epoch_bc) - np.min(epoch_bc)) * 1e6  # µV

        hep_times.append(p / sfreq)
        hep_pp.append(pp)

    return np.array(hep_times), np.array(hep_pp), n_rejected


# ── FUNCTION 4: PROCESS ONE SUBJECT ─────────────────────────────────────────
def process_subject(sub_id, data_dir="cap_data"):
    """
    Process one CAP subject. Returns per-stage summary DataFrame.
    Columns: sub, stage, pe_mean, pe_n, hep_pp_mean, hep_pp_n, hep_rejected
    """
    edf_path = Path(data_dir) / f"{sub_id}.edf"
    txt_path = Path(data_dir) / f"{sub_id}.txt"

    if not edf_path.exists() or not txt_path.exists():
        print(f"[SKIP] {sub_id}: files not found")
        return None

    # ── Load EDF ─────────────────────────────────────────────────────────────
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    except Exception as e:
        print(f"[SKIP] {sub_id}: EDF load error — {e}")
        return None

    sfreq = raw.info['sfreq']
    ch_names_upper = [c.upper() for c in raw.ch_names]

    # ── Find ECG channel ──────────────────────────────────────────────────────
    ecg_idx = next((i for i, c in enumerate(ch_names_upper)
                    if any(k in c for k in ['EKG', 'ECG', 'CARD'])), None)
    if ecg_idx is None:
        print(f"[SKIP] {sub_id}: no ECG channel")
        return None

    # ── Find frontal-central EEG channel ─────────────────────────────────────
    # Priority: F3-A2 or F4-A2, then F3-A1, F4-A1, then C3, C4
    fc_priority = ['F3-A2', 'F4-A2', 'F3-A1', 'F4-A1',
                   'F3', 'F4', 'C3-A2', 'C4-A2', 'C3', 'C4']
    eeg_idx = None
    eeg_name = None
    for target in fc_priority:
        for i, c in enumerate(ch_names_upper):
            if target in c and i != ecg_idx:
                eeg_idx = i
                eeg_name = raw.ch_names[i]
                break
        if eeg_idx is not None:
            break

    if eeg_idx is None:
        print(f"[SKIP] {sub_id}: no frontal-central EEG channel")
        return None

    ecg_signal = raw.get_data(picks=[ecg_idx])[0]
    eeg_signal = raw.get_data(picks=[eeg_idx])[0]

    print(f"[{sub_id}] ECG: {raw.ch_names[ecg_idx]} | "
          f"EEG: {eeg_name} | sfreq: {sfreq} Hz")

    # ── Parse staging ─────────────────────────────────────────────────────────
    df_stages = parse_staging(txt_path)
    if len(df_stages) == 0:
        print(f"[SKIP] {sub_id}: no valid stage epochs")
        return None

    # ── Pre-extract full HEP series ───────────────────────────────────────────
    hep_times, hep_pp, n_rej = extract_hep(ecg_signal, eeg_signal, sfreq)
    print(f"[{sub_id}] HEP: {len(hep_pp)} valid epochs, "
          f"{n_rej} rejected ({100*n_rej/(len(hep_pp)+n_rej+1e-9):.1f}%)")

    # ── Per-stage summaries ───────────────────────────────────────────────────
    results = []
    for stage in ['W', 'N1', 'N2', 'N3', 'R']:
        stage_epochs = df_stages[df_stages.stage == stage]
        if len(stage_epochs) == 0:
            continue

        pe_values = []
        hep_values = []

        for _, row in stage_epochs.iterrows():
            t_start = row['elapsed_sec']
            t_end = t_start + 30

            # Check for gap (epoch preceded by >30s gap = skip)
            idx = stage_epochs.index.get_loc(row.name)
            if idx > 0:
                prev_end = stage_epochs.iloc[idx - 1]['elapsed_sec'] + 30
                if t_start - prev_end > 35:
                    continue

            # EEG samples for this epoch
            s_start = int(t_start * sfreq)
            s_end = int(t_end * sfreq)
            if s_end > len(eeg_signal):
                continue

            eeg_epoch = eeg_signal[s_start:s_end] * 1e6  # convert to µV

            # Artefact check for PE
            if np.max(np.abs(eeg_epoch)) > 500:
                continue

            # Permutation entropy
            pe = permutation_entropy(eeg_epoch, order=PE_ORDER,
                                     delay=PE_DELAY, normalize=True)
            pe_values.append(pe)

            # HEP: find R-peaks within this epoch
            mask = (hep_times >= t_start) & (hep_times < t_end)
            if mask.sum() > 0:
                hep_values.extend(hep_pp[mask].tolist())

        # Apply minimum epoch threshold
        hep_ok = len(hep_values) >= MIN_HEP_EPOCHS

        results.append({
            'sub': sub_id,
            'stage': stage,
            'pe_mean': np.mean(pe_values) if pe_values else np.nan,
            'pe_n': len(pe_values),
            'hep_pp_mean': np.mean(hep_values) if hep_ok else np.nan,
            'hep_pp_n': len(hep_values),
            'hep_rejected': n_rej,
        })

        print(f"  {stage}: PE n={len(pe_values)} "
              f"mean={np.nanmean(pe_values):.4f} | "
              f"HEP n={len(hep_values)} "
              f"mean={np.nanmean(hep_values) if hep_values else float('nan'):.2f}µV")

    return pd.DataFrame(results)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    subjects = [f"n{i}" for i in range(1, 17)]
    all_results = []

    for sub in subjects:
        print(f"\n{'='*50}")
        result = process_subject(sub)
        if result is not None:
            all_results.append(result)

    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        df.to_csv("results_per_subject_stage.csv", index=False)
        print(f"\n{'='*50}")
        print(f"Saved: results_per_subject_stage.csv")
        print(f"Subjects processed: {df['sub'].nunique()}")
        print(f"\nStage counts:")
        print(df.groupby('stage')[['pe_mean', 'hep_pp_mean']].agg(
            ['mean', 'count']).round(4))
