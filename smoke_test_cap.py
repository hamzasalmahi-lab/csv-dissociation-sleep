import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import urllib.request
import re

# ── DOWNLOAD ONE HEALTHY SUBJECT ─────────────────────────────────────────────
BASE = "https://physionet.org/files/capslpdb/1.0.0"
Path("cap_data").mkdir(exist_ok=True)

for fname in ["n1.edf", "n1.txt"]:
    dest = Path(f"cap_data/{fname}")
    if not dest.exists():
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(f"{BASE}/{fname}", dest)
        print(f"  Done: {dest.stat().st_size/1e6:.1f} MB")
    else:
        print(f"  Already exists: {fname}")

# ── LOAD EDF ─────────────────────────────────────────────────────────────────
print("\nLoading EDF...")
raw = mne.io.read_raw_edf("cap_data/n1.edf", preload=True, verbose=False)
print(f"Channels ({len(raw.ch_names)}): {raw.ch_names}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]/3600:.2f} hours")

# ── FIND ECG AND FRONTAL-CENTRAL EEG ─────────────────────────────────────────
ecg_ch = [c for c in raw.ch_names
          if any(k in c.upper() for k in ['EKG', 'ECG', 'CARD'])]
eeg_fc = [c for c in raw.ch_names
          if any(k in c.upper() for k in ['F3', 'F4', 'C3', 'C4'])
          and 'EOG' not in c.upper()]

print(f"\nECG candidates: {ecg_ch}")
print(f"Frontal-central EEG candidates: {eeg_fc}")

# ── READ SLEEP STAGING ────────────────────────────────────────────────────────
print("\nReading sleep staging...")
stages = []
with open("cap_data/n1.txt", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            stages.append(line)

print(f"Annotation lines found: {len(stages)}")
print("First 5 lines:")
for s in stages[:5]:
    print(f"  {s}")

# ── PARSE STAGING ─────────────────────────────────────────────────────────────
stage_map = {'W': 0, 'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4, 'R': 5, 'MT': 6}
parsed = []
time_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")
for line in stages:
    parts = line.split()
    if len(parts) >= 2:
        stage = parts[0]
        if stage in stage_map:
            try:
                # CAP staging files may include multi-word position labels,
                # so find the first hh:mm:ss token instead of fixed index.
                time_str = next((p for p in parts if time_pattern.match(p)), None)
                if time_str is None:
                    continue
                h, m, s = map(int, time_str.split(':'))
                t_sec = h * 3600 + m * 60 + s
                parsed.append({'stage': stage, 'time_sec': t_sec,
                               'stage_num': stage_map[stage]})
            except Exception:
                pass

df_stages = pd.DataFrame(parsed)
print(f"\nParsed stage epochs: {len(df_stages)}")
print(df_stages['stage'].value_counts())

# ── ECG SIGNAL QUALITY CHECK ─────────────────────────────────────────────────
if ecg_ch:
    ecg_name = ecg_ch[0]
    sfreq = raw.info['sfreq']
    ecg_data = raw[ecg_name][0][0]

    # Plot first 60 seconds
    n60 = int(60 * sfreq)
    t60 = np.arange(n60) / sfreq
    ecg_60 = ecg_data[:n60]

    # Quick R-peak detection on 60s
    ecg_filt = mne.filter.filter_data(
        ecg_data[:int(300*sfreq)], sfreq, 5, 40, verbose=False)
    ecg_sq = ecg_filt ** 2
    peaks, _ = find_peaks(ecg_sq,
        distance=int(0.4*sfreq),
        height=np.percentile(ecg_sq, 90))
    hr = 60 / np.mean(np.diff(peaks[:20]) / sfreq) if len(peaks) > 5 else 0

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t60, ecg_60 * 1e6, lw=0.7, color='#028090')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("µV")
    ax.set_title(f"CAP n1 — ECG ({ecg_name}) — first 60 seconds  |  "
                 f"Est. HR: {hr:.0f} bpm")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("cap_smoke_ecg.png", dpi=150)
    print(f"\nECG channel: {ecg_name}")
    print(f"Estimated HR: {hr:.0f} bpm")
    print("ECG plot saved: cap_smoke_ecg.png")
else:
    print("\nNo ECG channel found — check channel names above")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n── SMOKE TEST SUMMARY ──")
print(f"EDF loads:        YES")
print(f"Channels:         {len(raw.ch_names)}")
print(f"Sfreq:            {raw.info['sfreq']} Hz")
print(f"Duration:         {raw.times[-1]/3600:.2f} h")
print(f"ECG present:      {'YES — ' + ecg_ch[0] if ecg_ch else 'NOT FOUND'}")
print(f"FC EEG channels:  {eeg_fc}")
print(f"Stage epochs:     {len(df_stages)}")
print(f"REM epochs:       {len(df_stages[df_stages.stage=='R'])}")
print(f"Wake epochs:      {len(df_stages[df_stages.stage=='W'])}")
print(f"N2 epochs:        {len(df_stages[df_stages.stage=='S2'])}")
print(f"N3 epochs:        {len(df_stages[df_stages.stage=='S3'])}")