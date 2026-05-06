"""
BRSM Project Report 2 — RQ4: Practice & Level Effects
======================================================
Formally characterises how RT changes across trials (lab)
and levels (phone) using linear regression.

Analyses performed
------------------
1. Group-level OLS regression: mean RT ~ trial/level number
   - Per cell (4 cells × 2 modalities)
   - Reports: slope (ms/trial), intercept, R², F, p-value

2. Per-participant regression slopes
   - One slope per participant per modality
   - Distribution of slopes: mean, SD, one-sample t-test vs 0
     (tests whether the GROUP shows a systematic trend)

3. Quadratic check
   - Tests whether a quadratic term improves fit
   - (captures learning plateau in lab / difficulty acceleration in phone)

Outputs
-------
  rq4_results.txt   — full regression report
  fig_rq4.png       — 4-panel figure (one per cell)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import pearsonr, ttest_1samp

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_all_data

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading data...")
df_trials, df_p = load_all_data(verbose=False)

R = []

def h(title, level=1):
    sep = '═' if level == 1 else '─'
    R.append(f"\n\n{sep*62}")
    R.append(title)
    R.append(f"{sep*62}")

def line(s=''):
    R.append(s)

# ══════════════════════════════════════════════════════════
# HELPER: OLS regression with full stats
# ══════════════════════════════════════════════════════════

def ols_stats(x, y):
    """
    Fit y = b0 + b1*x using OLS.
    Returns dict: slope, intercept, r, r2, F, p, se_slope, n
    """
    x, y  = np.array(x, dtype=float), np.array(y, dtype=float)
    n     = len(x)
    if n < 3:
        return None

    slope, intercept, r, p, se = stats.linregress(x, y)
    r2    = r**2
    df_reg, df_res = 1, n - 2
    F     = (r2 / df_reg) / ((1 - r2) / df_res) if (1 - r2) > 0 else np.nan

    return {
        'n': n, 'slope': slope, 'intercept': intercept,
        'r': r, 'r2': r2, 'F': F,
        'df_reg': df_reg, 'df_res': df_res,
        'p': p, 'se_slope': se,
    }


def quadratic_r2(x, y):
    """Fit quadratic and return R² improvement over linear."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(x) < 4:
        return np.nan, np.nan
    coeffs2 = np.polyfit(x, y, 2)
    y_hat2  = np.polyval(coeffs2, x)
    ss_res2 = np.sum((y - y_hat2)**2)
    ss_tot  = np.sum((y - np.mean(y))**2)
    r2_quad = 1 - ss_res2 / ss_tot if ss_tot > 0 else np.nan
    coeffs1 = np.polyfit(x, y, 1)
    y_hat1  = np.polyval(coeffs1, x)
    ss_res1 = np.sum((y - y_hat1)**2)
    r2_lin  = 1 - ss_res1 / ss_tot if ss_tot > 0 else np.nan
    return r2_lin, r2_quad


# ══════════════════════════════════════════════════════════
# 1. GROUP-LEVEL REGRESSION (mean RT per trial/level)
# ══════════════════════════════════════════════════════════

h("BRSM PROJECT REPORT 2 — RQ4: PRACTICE & LEVEL EFFECTS", level=1)
line("Formal OLS regression of RT on trial/level number.")
line("α = 0.05 (two-tailed)")

h("1. GROUP-LEVEL REGRESSION — Mean RT per trial/level number", level=1)
line("  Mean RT computed across all participants per trial/level.")
line()
line(f"  {'Cell':<20} {'n_pts':>6}  {'Slope':>10}  {'Intercept':>11}  "
     f"{'R²':>6}  {'F':>8}  {'p':>8}  Direction")
line(f"  {'─'*20}  {'─'*6}  {'─'*10}  {'─'*11}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*15}")

CELLS = [
    ('single',   'lab',   'Single/Lab'),
    ('single',   'phone', 'Single/Phone'),
    ('multiple', 'lab',   'Multiple/Lab'),
    ('multiple', 'phone', 'Multiple/Phone'),
]

group_reg = {}
group_means = {}   # store for plotting

for tload, mod, label in CELLS:
    mask = ((df_trials['target_load'] == tload) &
            (df_trials['modality']    == mod))
    grp  = df_trials[mask].copy()

    # Group mean RT per trial/level number
    gm = (grp.groupby('trial_num')['rt_first_ms']
              .mean()
              .reset_index()
              .sort_values('trial_num'))
    gm.columns = ['trial_num', 'mean_rt']
    group_means[label] = gm

    res = ols_stats(gm['trial_num'], gm['mean_rt'])
    if res is None:
        continue

    r2_lin, r2_quad = quadratic_r2(gm['trial_num'], gm['mean_rt'])
    direction = ('↑ increasing' if res['slope'] > 0 else '↓ decreasing')
    sig = ('***' if res['p'] < .001 else
           ('**'  if res['p'] < .01  else
            ('*'   if res['p'] < .05  else 'n.s.')))

    group_reg[label] = {**res, 'r2_quad': r2_quad, 'direction': direction}

    line(f"  {label:<20} {res['n']:>6}  {res['slope']:>+10.2f}  "
         f"{res['intercept']:>11.1f}  {res['r2']:>6.3f}  "
         f"{res['F']:>8.3f}  {res['p']:>8.4f}  {direction} {sig}")

line()
line("  Slope units: ms per trial (lab) or ms per level (phone)")
line("  Positive slope = RT increases (difficulty effect)")
line("  Negative slope = RT decreases (practice effect)")

# Quadratic improvement
h("1b. Quadratic vs Linear fit comparison", level=2)
line(f"  {'Cell':<20}  {'R²_linear':>10}  {'R²_quadratic':>13}  {'ΔR²':>7}  Interpretation")
line(f"  {'─'*20}  {'─'*10}  {'─'*13}  {'─'*7}  {'─'*25}")
for tload, mod, label in CELLS:
    if label not in group_reg:
        continue
    gm = group_means[label]
    r2_lin, r2_quad = quadratic_r2(gm['trial_num'], gm['mean_rt'])
    delta = r2_quad - r2_lin if not np.isnan(r2_quad) else np.nan
    interp = ('Quadratic better (plateau/acceleration)'
              if delta > 0.05 else 'Linear adequate')
    line(f"  {label:<20}  {r2_lin:>10.3f}  {r2_quad:>13.3f}  "
         f"{delta:>+7.3f}  {interp}")


# ══════════════════════════════════════════════════════════
# 2. PER-PARTICIPANT REGRESSION SLOPES
# ══════════════════════════════════════════════════════════

h("2. PER-PARTICIPANT SLOPES", level=1)
line("  One OLS slope fitted per participant.")
line("  One-sample t-test: H0: mean slope = 0 (no trend)")
line()
line(f"  {'Cell':<20}  {'n':>3}  {'M_slope':>9}  {'SD_slope':>9}  "
     f"{'t':>7}  {'df':>3}  {'p':>8}  Interpretation")
line(f"  {'─'*20}  {'─'*3}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*3}  {'─'*8}  {'─'*25}")

per_part_slopes = {}

for tload, mod, label in CELLS:
    mask = ((df_trials['target_load'] == tload) &
            (df_trials['modality']    == mod))
    grp  = df_trials[mask].copy()

    slopes = []
    for pid in grp['participant_id'].unique():
        p_data = grp[grp['participant_id'] == pid].sort_values('trial_num')
        rt     = p_data['rt_first_ms'].dropna().values
        tn     = p_data.loc[p_data['rt_first_ms'].notna(), 'trial_num'].values
        if len(rt) < 3:
            continue
        res = ols_stats(tn, rt)
        if res:
            slopes.append(res['slope'])

    slopes = np.array(slopes)
    per_part_slopes[label] = slopes

    if len(slopes) < 3:
        continue

    t, p  = ttest_1samp(slopes, 0)
    m, sd = np.mean(slopes), np.std(slopes, ddof=1)
    sig   = ('***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else 'n.s.')))
    direction = 'Significant ↑' if (p < .05 and m > 0) else \
                'Significant ↓' if (p < .05 and m < 0) else 'No trend'

    line(f"  {label:<20}  {len(slopes):>3}  {m:>+9.2f}  {sd:>9.2f}  "
         f"{t:>7.3f}  {len(slopes)-1:>3}  {p:>8.4f}  {direction} {sig}")

    line(f"    → Participants' RT changes by {m:+.1f} ms per trial/level on average")

line()
line("  Positive mean slope = group tends toward longer RT (difficulty)")
line("  Negative mean slope = group tends toward shorter RT (practice)")


# ══════════════════════════════════════════════════════════
# 3. SUMMARY INTERPRETATION
# ══════════════════════════════════════════════════════════

h("3. SUMMARY", level=1)
line("""
  Lab task (practice effect):
  ─────────────────────────
  Both lab conditions show negative slopes — RT decreases across trials,
  consistent with task familiarisation. The group-level regression
  captures the average learning curve. Participants begin slow and
  plateau to a stable performance level.

  Phone game (difficulty effect):
  ────────────────────────────────
  Single-target phone shows a positive slope — RT increases across
  levels as the game design deliberately increases difficulty (more
  distractors, larger arrays). Multiple-target phone shows a flatter
  trend — difficulty scaling is present but less pronounced, possibly
  because the first-touch RT metric is less sensitive to the number
  of targets.

  Quadratic fit:
  ──────────────
  If R²_quadratic >> R²_linear, the trend is non-linear (e.g.,
  rapid initial improvement that plateaus, or accelerating difficulty).
  This is particularly expected for the lab practice effect.
""")


# ══════════════════════════════════════════════════════════
# 4. SAVE TEXT REPORT
# ══════════════════════════════════════════════════════════

report_path = os.path.join(OUT_DIR, 'rq4_results.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(R))
print('\n'.join(R))
print(f"\nReport saved: {report_path}")


# ══════════════════════════════════════════════════════════
# 5. FIGURE — 4-panel (one per cell)
# ══════════════════════════════════════════════════════════

CELL_COLORS = {
    'Single/Lab'    : '#1976D2',
    'Single/Phone'  : '#42A5F5',
    'Multiple/Lab'  : '#D32F2F',
    'Multiple/Phone': '#EF9A9A',
}

fig = plt.figure(figsize=(14, 11))
fig.suptitle('RQ4 — Practice & Level Effects\nRT across Trials (Lab) and Levels (Phone)',
             fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
ax_map = {
    'Single/Lab'    : fig.add_subplot(gs[0, 0]),
    'Single/Phone'  : fig.add_subplot(gs[0, 1]),
    'Multiple/Lab'  : fig.add_subplot(gs[1, 0]),
    'Multiple/Phone': fig.add_subplot(gs[1, 1]),
}

for tload, mod, label in CELLS:
    ax    = ax_map[label]
    color = CELL_COLORS[label]
    mask  = ((df_trials['target_load'] == tload) &
             (df_trials['modality']    == mod))
    grp   = df_trials[mask].copy()
    gm    = group_means[label]
    res   = group_reg.get(label)

    # ── Individual participant lines (thin, transparent) ──
    for pid in grp['participant_id'].unique():
        p_data = (grp[grp['participant_id'] == pid]
                  .sort_values('trial_num')
                  .dropna(subset=['rt_first_ms']))
        ax.plot(p_data['trial_num'], p_data['rt_first_ms'],
                color=color, alpha=0.15, linewidth=0.8)

    # ── Group mean ± SE band ──
    se_per_trial = (grp.groupby('trial_num')['rt_first_ms']
                       .agg(['mean', 'std', 'count'])
                       .reset_index())
    se_per_trial['se'] = se_per_trial['std'] / np.sqrt(se_per_trial['count'])
    ax.fill_between(se_per_trial['trial_num'],
                    se_per_trial['mean'] - se_per_trial['se'],
                    se_per_trial['mean'] + se_per_trial['se'],
                    color=color, alpha=0.25)
    ax.plot(gm['trial_num'], gm['mean_rt'],
            color=color, linewidth=2.5, label='Group mean', zorder=4)

    # ── Linear regression line ──
    if res:
        x_fit = np.linspace(gm['trial_num'].min(), gm['trial_num'].max(), 100)
        y_fit = res['slope'] * x_fit + res['intercept']
        ls    = '--' if res['p'] < .05 else ':'
        ax.plot(x_fit, y_fit, ls, color='black', linewidth=1.8,
                label=f"Trend ({'p<.05' if res['p'] < .05 else 'n.s.'})")

        # ── Stats annotation ──
        sig = ('***' if res['p'] < .001 else
               ('**'  if res['p'] < .01  else
                ('*'   if res['p'] < .05  else 'n.s.')))
        stats_txt = (f"slope = {res['slope']:+.1f} ms/unit\n"
                     f"R² = {res['r2']:.3f},  {sig}")
        ax.text(0.97, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor=color, alpha=0.85))

    xlabel = 'Trial Number' if mod == 'lab' else 'Level'
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('RT — First Click (ms)', fontsize=9)
    ax.set_title(label, fontsize=10, fontweight='bold', color=color)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

fig_path = os.path.join(OUT_DIR, 'fig_rq4.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path}")
print("\nDone.")
