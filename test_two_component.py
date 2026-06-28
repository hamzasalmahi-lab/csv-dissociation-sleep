"""
Confirmatory tests on the two-component HEP coupling.

Pre-specified hypothesis (active inference framing of Gautier 2025):
    H_coupling-1   Coupling(N3)  > Coupling(R)    — Wilcoxon signed-rank, one-tailed
    H_coupling-2   Coupling(W)   > Coupling(R)    — Wilcoxon signed-rank, one-tailed
    H_coupling-3   Coupling(N3)  > Coupling(W)    — Wilcoxon signed-rank, two-tailed
                                                    (direction unclear a priori)

Effect size: r = Z / sqrt(N)  (Wilcoxon)
Reporting: Hedges' g for paired data as a secondary continuous effect-size measure.

Runs on results_two_component.csv produced by extract_two_component.py.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

CSV_PATH = "results_two_component.csv"


def wilcoxon_one_tailed(a, b, direction='greater'):
    """Wilcoxon signed-rank, one-tailed. Returns W, p, r, n."""
    a, b = np.asarray(a), np.asarray(b)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 4:
        return np.nan, np.nan, np.nan, len(a)
    res = stats.wilcoxon(a, b, alternative=direction, zero_method='wilcox')
    # Effect size r = |Z| / sqrt(N).  Recover Z from p via inverse normal.
    z = stats.norm.ppf(1 - res.pvalue) if res.pvalue < 1 else 0
    r_eff = abs(z) / np.sqrt(len(a))
    return res.statistic, res.pvalue, r_eff, len(a)


def hedges_g_paired(a, b):
    """Hedges' g for paired data, with small-sample correction."""
    a, b = np.asarray(a), np.asarray(b)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    n = len(d)
    if n < 2:
        return np.nan
    g = np.mean(d) / np.std(d, ddof=1)
    # Small-sample correction
    correction = 1 - 3 / (4 * (n - 1) - 1)
    return g * correction


def main():
    p = Path(CSV_PATH)
    if not p.exists():
        print(f"[ERROR] {CSV_PATH} not found. Run extract_two_component.py first.")
        return 1

    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Subjects: {df['sub'].nunique()}, "
          f"cross-channel: {df.groupby('sub')['cross_channel'].first().sum()}, "
          f"within-channel: {(~df.groupby('sub')['cross_channel'].first()).sum()}")

    # Pivot to per-subject per-stage table
    wide = df.pivot_table(
        index='sub', columns='stage',
        values='coupling_r', aggfunc='first')

    needed = ['W', 'N3', 'R']
    missing = [s for s in needed if s not in wide.columns]
    if missing:
        print(f"[WARN] missing stages in data: {missing}")

    print("\nPer-subject coupling table:")
    print(wide[needed].round(3).to_string())

    print("\nPer-stage descriptive statistics:")
    summary = wide[needed].agg(['mean', 'std', 'median', 'count']).round(3)
    print(summary.to_string())

    # ─── H_coupling-1: N3 > R ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("H1: Coupling(N3) > Coupling(R)  —  high B-precision tightens coupling")
    print("=" * 64)
    W, p_val, r, n = wilcoxon_one_tailed(wide['N3'], wide['R'], 'greater')
    g = hedges_g_paired(wide['N3'], wide['R'])
    print(f"Wilcoxon W={W:.1f}, p={p_val:.4f}, r={r:.3f}, n={n}, g={g:.3f}")
    print(f"Verdict: {'CONFIRMED' if p_val < 0.05 else 'NOT CONFIRMED'}")

    # ─── H_coupling-2: W > R ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("H2: Coupling(W) > Coupling(R)  —  REM lowest B-precision")
    print("=" * 64)
    W, p_val, r, n = wilcoxon_one_tailed(wide['W'], wide['R'], 'greater')
    g = hedges_g_paired(wide['W'], wide['R'])
    print(f"Wilcoxon W={W:.1f}, p={p_val:.4f}, r={r:.3f}, n={n}, g={g:.3f}")
    print(f"Verdict: {'CONFIRMED' if p_val < 0.05 else 'NOT CONFIRMED'}")

    # ─── H_coupling-3: N3 vs W ────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("H3: Coupling(N3) vs Coupling(W)  —  two-tailed (no directional prior)")
    print("=" * 64)
    a, b = wide['N3'].values, wide['W'].values
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) >= 4:
        res = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
        z = stats.norm.ppf(1 - res.pvalue / 2) if res.pvalue < 1 else 0
        r_eff = abs(z) / np.sqrt(len(a))
        g3 = hedges_g_paired(a, b)
        print(f"Wilcoxon W={res.statistic:.1f}, p={res.pvalue:.4f}, "
              f"r={r_eff:.3f}, n={len(a)}, g={g3:.3f}")
        print(f"Verdict: {'DIFFERENT' if res.pvalue < 0.05 else 'NOT DIFFERENT'}  "
              f"(N3 median = {np.median(a):.3f}, W median = {np.median(b):.3f})")
    else:
        print(f"[SKIP] only {len(a)} paired observations")

    # ─── Sensitivity: cross-channel subjects only ─────────────────────────
    cross_subs = df[df['cross_channel'] == True]['sub'].unique()
    if len(cross_subs) >= 4:
        print("\n" + "=" * 64)
        print(f"Sensitivity: cross-channel subjects only (n = {len(cross_subs)})")
        print("=" * 64)
        wide_cross = wide.loc[cross_subs]
        for label, (a_col, b_col, alt) in [
            ('N3 > R', ('N3', 'R',  'greater')),
            ('W  > R', ('W',  'R',  'greater')),
        ]:
            W, p_val, r_, n = wilcoxon_one_tailed(
                wide_cross[a_col], wide_cross[b_col], alt)
            print(f"{label}: W={W:.1f}, p={p_val:.4f}, r={r_:.3f}, n={n}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
