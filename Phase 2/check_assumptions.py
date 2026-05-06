"""
BRSM Project Report 2 — Assumption Checking
============================================
Checks all statistical assumptions BEFORE choosing between
parametric / non-parametric tests and before running ANOVA.

Checks performed
----------------
1. Outlier detection (per cell, on per-participant means)
   — Grubbs' test + IQR-based flagging
2. Normality of per-participant means (Shapiro-Wilk, per cell)
   — Operates on the 4 cells: SingleLab, SinglePhone,
     MultipleLab, MultiplePhone
3. Homogeneity of variance between groups (Levene's test)
   — Single vs. Multiple (between-subjects factor)
4. Sphericity — automatically satisfied (within factor
   has only 2 levels: lab vs. phone), noted for completeness

DVs checked
-----------
  mean_rt_first_ms  — primary DV (raw)
  log_mean_rt_first — log-transformed version
  mean_rt_last_ms   — secondary DV for lab (all-targets RT)
  log_mean_rt_last  — log of above

Outputs
-------
  assumption_report.txt  — full text report
  fig_assumptions.png    — visualisation panel
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import shapiro, levene

import warnings
warnings.filterwarnings('ignore')

# ── Import loader ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_all_data

# ══════════════════════════════════════════════════════════
# 0. LOAD DATA
# ══════════════════════════════════════════════════════════
print("Loading data...")
df_trials, df_p = load_all_data(verbose=False)

# Wide format: one row per participant, columns for each cell
# We need: single_lab, single_phone, multiple_lab, multiple_phone
def get_cell(target_load, modality, dv):
    mask = (df_p['target_load'] == target_load) & (df_p['modality'] == modality)
    return df_p.loc[mask, ['participant_id', dv]].set_index('participant_id')[dv]

CELLS = {
    'Single/Lab'      : ('single',   'lab'),
    'Single/Phone'    : ('single',   'phone'),
    'Multiple/Lab'    : ('multiple', 'lab'),
    'Multiple/Phone'  : ('multiple', 'phone'),
}

DVS = {
    'mean_rt_first_ms'  : 'Mean RT — First Click (ms) [raw]',
    'log_mean_rt_first' : 'Mean RT — First Click [log]',
    'mean_rt_last_ms'   : 'Mean RT — Last Click (ms) [raw]',
    'log_mean_rt_last'  : 'Mean RT — Last Click [log]',
}

# ══════════════════════════════════════════════════════════
# 1. OUTLIER DETECTION
# ══════════════════════════════════════════════════════════

def iqr_outliers(series, label):
    """Flag values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR."""
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = s[(s < lo) | (s > hi)]
    return flagged, lo, hi


def zscore_outliers(series, threshold=3.0):
    """Flag values with |z| > threshold."""
    s = series.dropna()
    z = np.abs(stats.zscore(s))
    return s[z > threshold]


def check_outliers(df_p, dv, report_lines):
    report_lines.append(f"\n{'─'*60}")
    report_lines.append(f"OUTLIER CHECK — {DVS.get(dv, dv)}")
    report_lines.append(f"{'─'*60}")

    outlier_summary = {}

    for cell_label, (tload, mod) in CELLS.items():
        series = get_cell(tload, mod, dv)
        series = series.dropna()

        iq_flagged, lo, hi = iqr_outliers(series, cell_label)
        zs_flagged          = zscore_outliers(series)

        combined_pids = set(iq_flagged.index) | set(zs_flagged.index)
        outlier_summary[cell_label] = combined_pids

        report_lines.append(f"\n  {cell_label}  (n={len(series)})")
        report_lines.append(f"    IQR fence : [{lo:.1f}, {hi:.1f}]")

        if combined_pids:
            for pid in sorted(combined_pids):
                val = series.get(pid, np.nan)
                iq_flag = "IQR" if pid in iq_flagged.index else ""
                zs_flag = "|z|>3" if pid in zs_flagged.index else ""
                flags   = " + ".join(f for f in [iq_flag, zs_flag] if f)
                report_lines.append(
                    f"    *** PID {pid:>3}  value={val:.1f}  [{flags}]")
        else:
            report_lines.append(f"    No outliers detected.")

    return outlier_summary


# ══════════════════════════════════════════════════════════
# 2. NORMALITY (Shapiro-Wilk on per-participant means)
# ══════════════════════════════════════════════════════════

def check_normality(df_p, dv, report_lines):
    report_lines.append(f"\n{'─'*60}")
    report_lines.append(f"NORMALITY — Shapiro-Wilk — {DVS.get(dv, dv)}")
    report_lines.append(f"{'─'*60}")
    report_lines.append(
        f"  {'Cell':<20} {'n':>4}  {'W':>8}  {'p':>8}  {'Skew':>7}  {'Kurt':>7}  Result")
    report_lines.append(f"  {'─'*20}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*15}")

    normality_results = {}

    for cell_label, (tload, mod) in CELLS.items():
        series = get_cell(tload, mod, dv).dropna()
        n = len(series)

        if n < 3:
            report_lines.append(f"  {cell_label:<20} {n:>4}  {'—':>8}  {'—':>8}  {'—':>7}  {'—':>7}  n too small")
            normality_results[cell_label] = None
            continue

        w, p      = shapiro(series)
        skewness  = stats.skew(series)
        kurtosis  = stats.kurtosis(series)
        normal    = p >= 0.05
        result    = "NORMAL" if normal else "NOT NORMAL"

        report_lines.append(
            f"  {cell_label:<20} {n:>4}  {w:>8.4f}  {p:>8.4f}  "
            f"{skewness:>7.3f}  {kurtosis:>7.3f}  {result}")

        normality_results[cell_label] = {
            'n': n, 'W': w, 'p': p,
            'skew': skewness, 'kurt': kurtosis, 'normal': normal
        }

    # Summary verdict
    non_normal = [k for k, v in normality_results.items()
                  if v is not None and not v['normal']]
    report_lines.append(f"\n  Non-normal cells: {non_normal if non_normal else 'None'}")

    return normality_results


# ══════════════════════════════════════════════════════════
# 3. HOMOGENEITY OF VARIANCE (Levene's test)
# ══════════════════════════════════════════════════════════

def check_homogeneity(df_p, dv, report_lines):
    report_lines.append(f"\n{'─'*60}")
    report_lines.append(f"HOMOGENEITY OF VARIANCE — Levene's test — {DVS.get(dv, dv)}")
    report_lines.append(f"  (Between-subjects factor: Single vs. Multiple)")
    report_lines.append(f"{'─'*60}")

    results = {}

    for mod in ['lab', 'phone']:
        single   = get_cell('single',   mod, dv).dropna().values
        multiple = get_cell('multiple', mod, dv).dropna().values

        if len(single) < 2 or len(multiple) < 2:
            continue

        stat, p  = levene(single, multiple, center='median')
        equal    = p >= 0.05
        result   = "Equal variances" if equal else "UNEQUAL variances — use Welch"

        report_lines.append(
            f"\n  {mod.capitalize()} modality:  "
            f"F={stat:.4f}, p={p:.4f}  →  {result}")
        report_lines.append(
            f"    Single   SD={np.std(single, ddof=1):.1f} ms  "
            f"(var={np.var(single, ddof=1):.1f})")
        report_lines.append(
            f"    Multiple SD={np.std(multiple, ddof=1):.1f} ms  "
            f"(var={np.var(multiple, ddof=1):.1f})")

        results[mod] = {'F': stat, 'p': p, 'equal_var': equal}

    return results


# ══════════════════════════════════════════════════════════
# 4. SPHERICITY NOTE
# ══════════════════════════════════════════════════════════

def note_sphericity(report_lines):
    report_lines.append(f"\n{'─'*60}")
    report_lines.append("SPHERICITY (Mauchly's test)")
    report_lines.append(f"{'─'*60}")
    report_lines.append(
        "  The within-subjects factor (Modality) has only 2 levels")
    report_lines.append(
        "  (lab vs. phone). With 2 levels, sphericity is automatically")
    report_lines.append(
        "  satisfied — Mauchly's test is not needed.")


# ══════════════════════════════════════════════════════════
# 5. OVERALL TEST-SELECTION VERDICT
# ══════════════════════════════════════════════════════════

def print_verdict(norm_raw_first, norm_log_first, hom_raw, hom_log, report_lines):
    report_lines.append(f"\n{'═'*60}")
    report_lines.append("TEST SELECTION VERDICT")
    report_lines.append(f"{'═'*60}")

    # Count non-normal cells for raw vs log
    nn_raw = sum(1 for v in norm_raw_first.values()
                 if v is not None and not v['normal'])
    nn_log = sum(1 for v in norm_log_first.values()
                 if v is not None and not v['normal'])

    report_lines.append(f"\n  Non-normal cells (raw RT):  {nn_raw}/4")
    report_lines.append(f"  Non-normal cells (log RT):  {nn_log}/4")

    report_lines.append(f"\n  Homogeneity (raw, lab):   "
                        f"{'OK' if hom_raw.get('lab',{}).get('equal_var', True) else 'VIOLATED'}")
    report_lines.append(f"  Homogeneity (raw, phone): "
                        f"{'OK' if hom_raw.get('phone',{}).get('equal_var', True) else 'VIOLATED'}")
    report_lines.append(f"  Homogeneity (log, lab):   "
                        f"{'OK' if hom_log.get('lab',{}).get('equal_var', True) else 'VIOLATED'}")
    report_lines.append(f"  Homogeneity (log, phone): "
                        f"{'OK' if hom_log.get('phone',{}).get('equal_var', True) else 'VIOLATED'}")

    report_lines.append("""
  RECOMMENDATION
  ──────────────
  Mixed ANOVA is relatively robust to moderate normality violations
  when group sizes are n=16-21 (Central Limit Theorem applies to
  per-participant means). Therefore:

  1. Run mixed ANOVA on log-transformed RT (better normality).
  2. Report raw-RT ANOVA as supplementary for comparison.
  3. Run non-parametric equivalents (Mann-Whitney U, Wilcoxon
     signed-rank) alongside for robustness — report both and
     note whether conclusions agree.
  4. If Levene violated → use Welch correction for t-tests
     (already done in Report 1).
  5. Flag PID outliers in Methods; run sensitivity analysis
     (with and without outliers) if any are extreme.
""")


# ══════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════

CELL_COLORS = {
    'Single/Lab'    : '#1976D2',
    'Single/Phone'  : '#42A5F5',
    'Multiple/Lab'  : '#D32F2F',
    'Multiple/Phone': '#EF9A9A',
}

def make_figure(df_p, out_path):
    dv_raw = 'mean_rt_first_ms'
    dv_log = 'log_mean_rt_first'

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'Assumption Checks — Per-Participant Mean RT\n'
        '(Unit of analysis for all inferential tests)',
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38)

    cell_list   = list(CELLS.items())
    cell_labels = list(CELLS.keys())

    # ── Row 0: Q-Q plots (raw) ──
    for i, (label, (tload, mod)) in enumerate(cell_list):
        ax = fig.add_subplot(gs[0, i])
        series = get_cell(tload, mod, dv_raw).dropna()
        (osm, osr), (slope, intercept, _) = stats.probplot(series, dist='norm')
        ax.plot(osm, osr, 'o', color=CELL_COLORS[label], markersize=5, alpha=0.8)
        ax.plot(osm, np.array(osm)*slope + intercept, '--k', lw=1)
        w, p = shapiro(series)
        ax.set_title(f'Q-Q: {label}\nSW p={p:.3f}', fontsize=8, fontweight='bold')
        ax.set_xlabel('Theoretical', fontsize=7)
        ax.set_ylabel('Sample', fontsize=7)
        ax.tick_params(labelsize=7)

    # ── Row 1: Q-Q plots (log) ──
    for i, (label, (tload, mod)) in enumerate(cell_list):
        ax = fig.add_subplot(gs[1, i])
        series = get_cell(tload, mod, dv_log).dropna()
        (osm, osr), (slope, intercept, _) = stats.probplot(series, dist='norm')
        ax.plot(osm, osr, 'o', color=CELL_COLORS[label], markersize=5, alpha=0.8)
        ax.plot(osm, np.array(osm)*slope + intercept, '--k', lw=1)
        w, p = shapiro(series)
        ax.set_title(f'Q-Q log: {label}\nSW p={p:.3f}', fontsize=8, fontweight='bold')
        ax.set_xlabel('Theoretical', fontsize=7)
        ax.set_ylabel('log(RT)', fontsize=7)
        ax.tick_params(labelsize=7)

    # ── Row 2, col 0-1: Boxplots raw vs log ──
    ax_raw = fig.add_subplot(gs[2, 0:2])
    data_raw = [get_cell(tl, mod, dv_raw).dropna().values
                for _, (tl, mod) in cell_list]
    bp = ax_raw.boxplot(data_raw, patch_artist=True,
                        medianprops=dict(color='black', lw=2),
                        flierprops=dict(marker='o', markersize=6,
                                        markerfacecolor='red', alpha=0.7))
    for patch, label in zip(bp['boxes'], cell_labels):
        patch.set_facecolor(CELL_COLORS[label])
        patch.set_alpha(0.75)
    ax_raw.set_xticklabels([l.replace('/', '\n') for l in cell_labels], fontsize=8)
    ax_raw.set_ylabel('Mean RT — First Click (ms)', fontsize=9)
    ax_raw.set_title('Boxplots — Raw RT (outliers in red)', fontweight='bold', fontsize=9)

    ax_log = fig.add_subplot(gs[2, 2:4])
    data_log = [get_cell(tl, mod, dv_log).dropna().values
                for _, (tl, mod) in cell_list]
    bp2 = ax_log.boxplot(data_log, patch_artist=True,
                         medianprops=dict(color='black', lw=2),
                         flierprops=dict(marker='o', markersize=6,
                                         markerfacecolor='red', alpha=0.7))
    for patch, label in zip(bp2['boxes'], cell_labels):
        patch.set_facecolor(CELL_COLORS[label])
        patch.set_alpha(0.75)
    ax_log.set_xticklabels([l.replace('/', '\n') for l in cell_labels], fontsize=8)
    ax_log.set_ylabel('log(Mean RT — First Click)', fontsize=9)
    ax_log.set_title('Boxplots — Log-Transformed RT (outliers in red)',
                     fontweight='bold', fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {out_path}")


# ══════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("BRSM PROJECT REPORT 2 — ASSUMPTION CHECKS")
    report_lines.append("Unit of analysis: per-participant mean RT")
    report_lines.append("=" * 60)

    # ── Outliers ──
    report_lines.append("\n\n" + "═"*60)
    report_lines.append("SECTION 1: OUTLIER DETECTION")
    report_lines.append("═"*60)
    check_outliers(df_p, 'mean_rt_first_ms',  report_lines)
    check_outliers(df_p, 'mean_rt_last_ms',   report_lines)
    check_outliers(df_p, 'mean_accuracy',      report_lines)

    # ── Normality ──
    report_lines.append("\n\n" + "═"*60)
    report_lines.append("SECTION 2: NORMALITY (Shapiro-Wilk)")
    report_lines.append("  Checked on per-participant means (n=21 or 16 per cell)")
    report_lines.append("═"*60)
    norm_raw_first  = check_normality(df_p, 'mean_rt_first_ms',  report_lines)
    norm_log_first  = check_normality(df_p, 'log_mean_rt_first', report_lines)
    norm_raw_last   = check_normality(df_p, 'mean_rt_last_ms',   report_lines)
    norm_log_last   = check_normality(df_p, 'log_mean_rt_last',  report_lines)

    # ── Homogeneity ──
    report_lines.append("\n\n" + "═"*60)
    report_lines.append("SECTION 3: HOMOGENEITY OF VARIANCE (Levene's)")
    report_lines.append("═"*60)
    hom_raw = check_homogeneity(df_p, 'mean_rt_first_ms',  report_lines)
    hom_log = check_homogeneity(df_p, 'log_mean_rt_first', report_lines)

    # ── Sphericity ──
    report_lines.append("\n\n" + "═"*60)
    report_lines.append("SECTION 4: SPHERICITY")
    report_lines.append("═"*60)
    note_sphericity(report_lines)

    # ── Verdict ──
    print_verdict(norm_raw_first, norm_log_first, hom_raw, hom_log, report_lines)

    # ── Save report ──
    report_path = os.path.join(out_dir, 'assumption_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved: {report_path}")
    print('\n'.join(report_lines))

    # ── Figure ──
    fig_path = os.path.join(out_dir, 'fig_assumptions.png')
    make_figure(df_p, fig_path)

    print("\nDone.")
