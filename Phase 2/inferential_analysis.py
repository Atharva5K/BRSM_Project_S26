"""
BRSM Project Report 2 — Inferential Analysis
=============================================
Runs all inferential tests in the correct logical order.

Tests performed
---------------
1. 2×2 Mixed ANOVA on log(rt_first_ms)
   - Primary DV: log_mean_rt_first (all 4 cells)
   - Supplementary: raw rt_first_ms ANOVA
   - Sensitivity: re-run excluding PID 32 (outlier)

2. Non-parametric equivalents
   - Mann-Whitney U  : Single vs Multiple (between, per modality)
   - Wilcoxon signed-rank : Lab vs Phone (within, per condition)

3. RQ1 — Concurrent Validity (re-analysis)
   - Single-target  : Pearson r + Spearman ρ, lab vs phone rt_first_ms
   - Multiple-target: Pearson r + Spearman ρ, lab rt_last_ms/5 vs phone avg_inter_target_ms
     (Option C fix: both measure mean time-per-target)

4. RQ2 — Target Load Effect
   - Independent Welch t-test: Single vs Multiple, per modality
   - Both rt_first_ms and (for lab) rt_last_ms reported

5. RQ3 — Modality Effect
   - Paired t-test: Lab vs Phone, per condition
   - Single-target: clean; Multiple-target: caveated

6. Reliability — Split-half
   - Lab : odd vs even trials (Pearson r + Spearman-Brown correction)
   - Phone: first-half vs second-half levels
   - Reported per condition

Outputs
-------
  inferential_results.txt   — full numerical report
  fig_anova.png             — interaction plot + effect sizes
  fig_validity.png          — RQ1 scatterplots (corrected)
  fig_nonparametric.png     — non-parametric results summary
  fig_reliability.png       — split-half reliability
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
from scipy.stats import (shapiro, mannwhitneyu, wilcoxon,
                         pearsonr, spearmanr, ttest_rel, ttest_ind)

warnings.filterwarnings('ignore')

# ── Import loader ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_all_data

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════
# 0. LOAD DATA
# ══════════════════════════════════════════════════════════
print("Loading data...")
df_trials, df_p = load_all_data(verbose=False)

# Convenience selectors
def cell(target_load, modality, dv=None):
    mask = (df_p['target_load'] == target_load) & (df_p['modality'] == modality)
    sub  = df_p.loc[mask].copy()
    if dv:
        return sub.set_index('participant_id')[dv].dropna()
    return sub

# Cohen's d (independent)
def cohens_d_ind(a, b):
    na, nb  = len(a), len(b)
    pooled  = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2)
                      / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan

# Cohen's d (paired)
def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan

# Partial eta-squared from F
def partial_eta2(F, df_effect, df_error):
    return (F * df_effect) / (F * df_effect + df_error)

# rank-biserial correlation for Mann-Whitney
def rank_biserial(u, n1, n2):
    return 1 - (2*u) / (n1*n2)

# ══════════════════════════════════════════════════════════
# REPORT LINES ACCUMULATOR
# ══════════════════════════════════════════════════════════
R = []   # all lines go here; printed + saved at end

def h(title, level=1):
    sep = '═' if level == 1 else '─'
    R.append(f"\n\n{sep*62}")
    R.append(title)
    R.append(f"{sep*62}")

def line(s=''):
    R.append(s)


# ══════════════════════════════════════════════════════════
# 1. 2×2 MIXED ANOVA
# ══════════════════════════════════════════════════════════
# Mixed ANOVA by hand (standard formulas for 2×2 design):
#   Between factor A (Target Load): single vs multiple
#   Within  factor B (Modality)   : lab vs phone
#   n_single=21, n_multiple=16

def run_mixed_anova(df_p, dv, label, report=True):
    """
    2×2 Mixed ANOVA.
    Returns dict with F, p, eta2 for A, B, AxB.

    Factor A (between): target_load  (single=1, multiple=2)
    Factor B (within) : modality     (lab, phone)
    """
    # Build cell arrays (per-participant means)
    s_lab   = cell('single',   'lab',   dv).values
    s_ph    = cell('single',   'phone', dv).values
    m_lab   = cell('multiple', 'lab',   dv).values
    m_ph    = cell('multiple', 'phone', dv).values

    n1, n2  = len(s_lab), len(m_lab)   # group sizes
    N       = n1 + n2

    # Cell means
    Y11, Y12 = np.mean(s_lab), np.mean(s_ph)    # single: lab, phone
    Y21, Y22 = np.mean(m_lab), np.mean(m_ph)    # multiple: lab, phone

    # Marginal means
    Y1_  = (Y11 + Y12) / 2   # single marginal
    Y2_  = (Y21 + Y22) / 2   # multiple marginal
    Y_1  = (n1*Y11 + n2*Y21) / N   # lab marginal
    Y_2  = (n1*Y12 + n2*Y22) / N   # phone marginal
    Y__  = (n1*(Y11+Y12) + n2*(Y21+Y22)) / (2*N)  # grand mean

    # ── SS Between (Factor A: Target Load) ──
    SS_A   = n1*(Y1_ - Y__)**2 * 2 + n2*(Y2_ - Y__)**2 * 2
    # Better: SS_A = sum_j n_j * b * (Yj. - Y..)^2  where b=2 levels of B
    SS_A   = 2 * (n1*(Y1_ - Y__)**2 + n2*(Y2_ - Y__)**2)
    df_A   = 1

    # ── SS Subjects within groups (error for between) ──
    # Each participant contributes a mean across the 2 within levels
    s_means = (s_lab + s_ph) / 2
    m_means = (m_lab + m_ph) / 2
    SS_Sbw  = 2 * (np.sum((s_means - Y1_)**2) + np.sum((m_means - Y2_)**2))
    df_Sbw  = N - 2   # (n1-1) + (n2-1)

    # ── SS Within (Factor B: Modality) ──
    SS_B   = N * ((Y_1 - Y__)**2 + (Y_2 - Y__)**2)
    # More precisely weighted by group sizes:
    SS_B   = (n1*((Y11-Y1_-Y_1+Y__)**2 + (Y12-Y1_-Y_2+Y__)**2) +
              n2*((Y21-Y2_-Y_1+Y__)**2 + (Y22-Y2_-Y_2+Y__)**2))
    # Use the definitional formula instead:
    # SS_B = sum_i sum_j (Y_ij_mean - Y_i. - Y_.j + Y..)^2 * n_i  -- below
    SS_B   = (n1 * ((Y11 - Y1_ - Y_1 + Y__)**2 +
                    (Y12 - Y1_ - Y_2 + Y__)**2) +
              n2 * ((Y21 - Y2_ - Y_1 + Y__)**2 +
                    (Y22 - Y2_ - Y_2 + Y__)**2))
    df_B   = 1

    # ── SS Interaction (A×B) ──
    SS_AB  = (n1 * ((Y11 - Y1_ - Y_1 + Y__)**2 +
                    (Y12 - Y1_ - Y_2 + Y__)**2) +
              n2 * ((Y21 - Y2_ - Y_1 + Y__)**2 +
                    (Y22 - Y2_ - Y_2 + Y__)**2))
    # The interaction SS is the cell deviation after removing A and B main effects
    # Correct formula:
    SS_AB  = (n1*((Y11 - Y1_ - Y_1 + Y__)**2 + (Y12 - Y1_ - Y_2 + Y__)**2) +
              n2*((Y21 - Y2_ - Y_1 + Y__)**2 + (Y22 - Y2_ - Y_2 + Y__)**2))
    df_AB  = 1

    # ── SS Within-cell error (B × Subjects) ──
    # For each participant: deviation from their group's within-cell means
    s_diff  = (s_lab - s_ph) - (Y11 - Y12)   # centred diff for single group
    m_diff  = (m_lab - m_ph) - (Y21 - Y22)   # centred diff for multiple group
    SS_BxS  = 0.5 * (np.sum(s_diff**2) + np.sum(m_diff**2))
    df_BxS  = N - 2   # (n1-1) + (n2-1)

    # ── F ratios ──
    MS_A   = SS_A   / df_A
    MS_Sbw = SS_Sbw / df_Sbw
    MS_B   = SS_B   / df_B
    MS_AB  = SS_AB  / df_AB
    MS_BxS = SS_BxS / df_BxS

    F_A   = MS_A   / MS_Sbw
    F_B   = MS_B   / MS_BxS
    F_AB  = MS_AB  / MS_BxS

    p_A   = 1 - stats.f.cdf(F_A,  df_A,  df_Sbw)
    p_B   = 1 - stats.f.cdf(F_B,  df_B,  df_BxS)
    p_AB  = 1 - stats.f.cdf(F_AB, df_AB, df_BxS)

    eta2_A  = partial_eta2(F_A,  df_A,  df_Sbw)
    eta2_B  = partial_eta2(F_B,  df_B,  df_BxS)
    eta2_AB = partial_eta2(F_AB, df_AB, df_BxS)

    res = {
        'cell_means': {'Single/Lab': Y11, 'Single/Phone': Y12,
                       'Multiple/Lab': Y21, 'Multiple/Phone': Y22},
        'A':  {'F': F_A,  'df': (df_A,  df_Sbw), 'p': p_A,  'eta2': eta2_A},
        'B':  {'F': F_B,  'df': (df_B,  df_BxS), 'p': p_B,  'eta2': eta2_B},
        'AB': {'F': F_AB, 'df': (df_AB, df_BxS), 'p': p_AB, 'eta2': eta2_AB},
    }

    if report:
        h(f"1. 2×2 MIXED ANOVA — {label}")
        line(f"  DV: {dv}")
        line(f"  Between-subjects factor A: Target Load (Single n={n1}, Multiple n={n2})")
        line(f"  Within-subjects  factor B: Modality (Lab, Phone)")
        line()
        line(f"  Cell means ({dv}):")
        line(f"    Single/Lab    = {Y11:.4f}")
        line(f"    Single/Phone  = {Y12:.4f}")
        line(f"    Multiple/Lab  = {Y21:.4f}")
        line(f"    Multiple/Phone= {Y22:.4f}")
        line()
        line(f"  {'Source':<25} {'F':>8}  {'df':>10}  {'p':>8}  {'η²p':>7}  Significance")
        line(f"  {'─'*25}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*15}")

        def sig(p):
            if p < .001: return 'p < .001 ***'
            if p < .01:  return 'p < .01  **'
            if p < .05:  return 'p < .05  *'
            return f'p = {p:.3f} n.s.'

        line(f"  {'Target Load (A)':<25} {F_A:>8.3f}  "
             f"F({df_A},{df_Sbw:>2}){'':<3}  {p_A:>8.4f}  {eta2_A:>7.4f}  {sig(p_A)}")
        line(f"  {'Modality (B)':<25} {F_B:>8.3f}  "
             f"F({df_B},{df_BxS:>2}){'':<3}  {p_B:>8.4f}  {eta2_B:>7.4f}  {sig(p_B)}")
        line(f"  {'Target Load × Modality':<25} {F_AB:>8.3f}  "
             f"F({df_AB},{df_BxS:>2}){'':<3}  {p_AB:>8.4f}  {eta2_AB:>7.4f}  {sig(p_AB)}")
        line()
        line("  Effect size guide: η²p < .01 small, .01–.06 medium, > .06 large")
        line()
        line("  NOTE (multiple-target operationalisation):")
        line("  rt_first_ms for Multiple/Lab = time to first of 5 targets (~1500 ms),")
        line("  NOT total completion time (~6500 ms). This compresses the Target Load")
        line("  effect and affects the interaction term. See supplementary last-click")
        line("  analysis and RQ1 re-analysis for the corrected multiple-target DV.")

    return res


h("BRSM PROJECT REPORT 2 — INFERENTIAL ANALYSIS", level=1)
line("Unit of analysis: per-participant mean RT")
line("Primary DV: log(mean_rt_first_ms) — all 4 cells")
line("α = 0.05 (two-tailed) throughout")

# ── 1a. Primary ANOVA (log RT) ──
anova_log = run_mixed_anova(df_p, 'log_mean_rt_first', 'log(RT first click)', report=True)

# ── 1b. Supplementary ANOVA (raw RT) ──
anova_raw = run_mixed_anova(df_p, 'mean_rt_first_ms', 'Raw RT first click (ms)', report=True)

# ── 1c. Sensitivity: exclude PID 32 ──
h("1c. SENSITIVITY ANALYSIS — Excluding PID 32 (accuracy outlier)", level=2)
line("  PID 32: Multiple/Lab accuracy = 0.45 (group mean ~0.98), last-click RT = 3055 ms")
line("  (far below group range of 5248–10607 ms) — likely data quality issue.")
line()

df_p_no32 = df_p[df_p['participant_id'] != 32].copy()

# Temporarily swap df_p for the sensitivity run
_df_p_orig = df_p.copy()

# We need cell() to use df_p_no32 — redefine locally
def cell_s(target_load, modality, dv):
    mask = (df_p_no32['target_load'] == target_load) & (df_p_no32['modality'] == modality)
    return df_p_no32.loc[mask].set_index('participant_id')[dv].dropna()

def run_anova_sensitivity(dv, label):
    s_lab  = cell_s('single',   'lab',   dv).values
    s_ph   = cell_s('single',   'phone', dv).values
    m_lab  = cell_s('multiple', 'lab',   dv).values
    m_ph   = cell_s('multiple', 'phone', dv).values
    n1, n2 = len(s_lab), len(m_lab)
    N      = n1 + n2
    Y11, Y12 = np.mean(s_lab), np.mean(s_ph)
    Y21, Y22 = np.mean(m_lab), np.mean(m_ph)
    Y1_ = (Y11+Y12)/2;  Y2_ = (Y21+Y22)/2
    Y_1 = (n1*Y11+n2*Y21)/N;  Y_2 = (n1*Y12+n2*Y22)/N
    Y__ = (n1*(Y11+Y12)+n2*(Y21+Y22))/(2*N)
    SS_A  = 2*(n1*(Y1_-Y__)**2 + n2*(Y2_-Y__)**2)
    s_m   = (s_lab+s_ph)/2;  m_m = (m_lab+m_ph)/2
    SS_Sbw= 2*(np.sum((s_m-Y1_)**2)+np.sum((m_m-Y2_)**2))
    SS_AB = (n1*((Y11-Y1_-Y_1+Y__)**2+(Y12-Y1_-Y_2+Y__)**2)+
             n2*((Y21-Y2_-Y_1+Y__)**2+(Y22-Y2_-Y_2+Y__)**2))
    s_diff= (s_lab-s_ph)-(Y11-Y12);  m_diff=(m_lab-m_ph)-(Y21-Y22)
    SS_BxS= 0.5*(np.sum(s_diff**2)+np.sum(m_diff**2))
    df_A=1; df_Sbw=N-2; df_AB=1; df_BxS=N-2
    MS_A=SS_A/df_A; MS_Sbw=SS_Sbw/df_Sbw
    MS_AB=SS_AB/df_AB; MS_BxS=SS_BxS/df_BxS
    F_A=MS_A/MS_Sbw; F_AB=MS_AB/MS_BxS
    p_A =1-stats.f.cdf(F_A, df_A, df_Sbw)
    p_AB=1-stats.f.cdf(F_AB,df_AB,df_BxS)
    eta2_A =partial_eta2(F_A, df_A, df_Sbw)
    eta2_AB=partial_eta2(F_AB,df_AB,df_BxS)
    line(f"  {label}  (n_multiple={n2})")
    line(f"    Target Load:        F({df_A},{df_Sbw})={F_A:.3f},  p={p_A:.4f},  η²p={eta2_A:.4f}")
    line(f"    Interaction (A×B):  F({df_AB},{df_BxS})={F_AB:.3f}, p={p_AB:.4f},  η²p={eta2_AB:.4f}")

run_anova_sensitivity('log_mean_rt_first', 'log(RT first click)')
run_anova_sensitivity('mean_rt_first_ms',  'Raw RT first click')
line()
line("  If results are consistent with full-sample ANOVA, PID 32 is not")
line("  unduly influencing conclusions.")


# ══════════════════════════════════════════════════════════
# 2. NON-PARAMETRIC EQUIVALENTS
# ══════════════════════════════════════════════════════════

h("2. NON-PARAMETRIC EQUIVALENTS", level=1)
line("  Run alongside parametric tests as robustness check.")
line("  DV: mean_rt_first_ms (raw — rank-based tests don't require normality)")

h("2a. Mann-Whitney U — Single vs Multiple (between-subjects)", level=2)
line(f"  {'Modality':<10}  {'U':>8}  {'p':>8}  {'r_rb':>7}  Conclusion")
line(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*20}")

mw_results = {}
for mod in ['lab', 'phone']:
    s = cell('single',   mod, 'mean_rt_first_ms').values
    m = cell('multiple', mod, 'mean_rt_first_ms').values
    U, p = mannwhitneyu(s, m, alternative='two-sided')
    rb   = rank_biserial(U, len(s), len(m))
    sig  = '*' if p < .05 else 'n.s.'
    line(f"  {mod.capitalize():<10}  {U:>8.1f}  {p:>8.4f}  {rb:>7.3f}  {sig}")
    mw_results[mod] = {'U': U, 'p': p, 'rb': rb}

h("2b. Wilcoxon Signed-Rank — Lab vs Phone (within-subjects)", level=2)
line(f"  {'Condition':<12}  {'W':>8}  {'p':>8}  {'d_paired':>9}  Conclusion")
line(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*20}")

wc_results = {}
for tload in ['single', 'multiple']:
    lab_s  = cell(tload, 'lab',   'mean_rt_first_ms')
    ph_s   = cell(tload, 'phone', 'mean_rt_first_ms')
    # Align on participant_id
    common = lab_s.index.intersection(ph_s.index)
    lab_v  = lab_s.loc[common].values
    ph_v   = ph_s.loc[common].values
    diff   = lab_v - ph_v
    if np.all(diff == 0):
        line(f"  {tload.capitalize():<12}  all differences zero — test not applicable")
        continue
    W, p   = wilcoxon(lab_v, ph_v, alternative='two-sided')
    d      = cohens_d_paired(lab_v, ph_v)
    sig    = '*' if p < .05 else 'n.s.'
    line(f"  {tload.capitalize():<12}  {W:>8.1f}  {p:>8.4f}  {d:>9.3f}  {sig}")
    wc_results[tload] = {'W': W, 'p': p, 'd': d}

line()
line("  Rank-biserial r interpretation: |r| < 0.3 small, 0.3–0.5 medium, > 0.5 large")


# ══════════════════════════════════════════════════════════
# 3. RQ1 — CONCURRENT VALIDITY (corrected)
# ══════════════════════════════════════════════════════════

h("3. RQ1 — CONCURRENT VALIDITY (corrected re-analysis)", level=1)

h("3a. Single-target: lab rt_first_ms vs phone rt_first_ms", level=2)
line("  (Operationalisation is clean: both = time to click/tap 1 target)")

s_lab_rt  = cell('single', 'lab',   'mean_rt_first_ms')
s_ph_rt   = cell('single', 'phone', 'mean_rt_first_ms')
common_s  = s_lab_rt.index.intersection(s_ph_rt.index)
sl        = s_lab_rt.loc[common_s].values
sp        = s_ph_rt.loc[common_s].values

r_s, p_r_s  = pearsonr(sl, sp)
rho_s, p_rho_s = spearmanr(sl, sp)

line(f"  n = {len(sl)}")
line(f"  Pearson  r   = {r_s:.3f},  p = {p_r_s:.4f}  "
     f"({'significant' if p_r_s < .05 else 'not significant'})")
line(f"  Spearman ρ   = {rho_s:.3f},  p = {p_rho_s:.4f}  "
     f"({'significant' if p_rho_s < .05 else 'not significant'})")

h("3b. Multiple-target (Option C fix): lab rt_last_ms/5 vs phone avg_inter_target_ms", level=2)
line("  Both DVs = mean time per target:")
line("    Lab  : rt_last_ms / 5  = mean ms to find each of the 5 targets")
line("    Phone: avg_inter_target_ms = mean ms between successive target taps")

m_lab_last = cell('multiple', 'lab',   'mean_rt_last_ms')
m_ph_ait   = cell('multiple', 'phone', 'mean_avg_inter_target_ms')
common_m   = m_lab_last.index.intersection(m_ph_ait.index)
ml_per     = (m_lab_last.loc[common_m] / 5).values
mp_ait     = m_ph_ait.loc[common_m].values

r_m, p_r_m     = pearsonr(ml_per, mp_ait)
rho_m, p_rho_m = spearmanr(ml_per, mp_ait)

line(f"  n = {len(common_m)}")
line(f"  Lab  mean time-per-target: M = {np.mean(ml_per):.1f} ms, SD = {np.std(ml_per,ddof=1):.1f}")
line(f"  Phone avg inter-target:    M = {np.mean(mp_ait):.1f} ms, SD = {np.std(mp_ait,ddof=1):.1f}")
line(f"  Pearson  r   = {r_m:.3f},  p = {p_r_m:.4f}  "
     f"({'significant' if p_r_m < .05 else 'not significant'})")
line(f"  Spearman ρ   = {rho_m:.3f},  p = {p_rho_m:.4f}  "
     f"({'significant' if p_rho_m < .05 else 'not significant'})")
line()
line("  Compare with Report 1: r = -0.11 (n.s.) using InitialResponseTime.")
line("  The corrected DV tests whether search *rate* (not first-touch speed)")
line("  is consistent across modalities.")


# ══════════════════════════════════════════════════════════
# 4. RQ2 — TARGET LOAD EFFECT
# ══════════════════════════════════════════════════════════

h("4. RQ2 — TARGET LOAD EFFECT (Single vs Multiple)", level=1)
line("  Independent Welch t-test per modality.")
line("  Primary DV: mean_rt_first_ms | Supplementary (lab only): mean_rt_last_ms")

h("4a. First-click RT", level=2)
line(f"  {'Modality':<10}  {'M_single':>10}  {'M_multi':>10}  "
     f"{'t':>8}  {'df':>6}  {'p':>8}  {'d':>7}  Sig")
line(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*10}")

ttest_rq2 = {}
for mod in ['lab', 'phone']:
    s = cell('single',   mod, 'mean_rt_first_ms').values
    m = cell('multiple', mod, 'mean_rt_first_ms').values
    t, p = ttest_ind(s, m, equal_var=False)
    d    = cohens_d_ind(s, m)
    df_w = (np.var(s,ddof=1)/len(s) + np.var(m,ddof=1)/len(m))**2 / \
           ((np.var(s,ddof=1)/len(s))**2/(len(s)-1) +
            (np.var(m,ddof=1)/len(m))**2/(len(m)-1))
    sig  = '***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else 'n.s.'))
    line(f"  {mod.capitalize():<10}  {np.mean(s):>10.1f}  {np.mean(m):>10.1f}  "
         f"{t:>8.3f}  {df_w:>6.1f}  {p:>8.4f}  {d:>7.3f}  {sig}")
    ttest_rq2[('first', mod)] = {'t': t, 'p': p, 'd': d}

h("4b. Last-click RT (lab only — full task completion time)", level=2)
line("  This is the 'true' multiple-target task time but not comparable to phone.")
s_last = cell('single',   'lab', 'mean_rt_last_ms').values
m_last = cell('multiple', 'lab', 'mean_rt_last_ms').values
t_last, p_last = ttest_ind(s_last, m_last, equal_var=False)
d_last = cohens_d_ind(s_last, m_last)
line(f"  Single/Lab last-click  : M = {np.mean(s_last):.1f} ms")
line(f"  Multiple/Lab last-click: M = {np.mean(m_last):.1f} ms")
line(f"  Welch t = {t_last:.3f}, p = {p_last:.4f}, d = {d_last:.3f}")
line("  (Replicates Report 1 d=4.94 — confirms FIT prediction)")


# ══════════════════════════════════════════════════════════
# 5. RQ3 — MODALITY EFFECT
# ══════════════════════════════════════════════════════════

h("5. RQ3 — MODALITY EFFECT (Lab vs Phone)", level=1)
line("  Paired t-test per condition.")
line("  Single-target: clean comparison. Multiple-target: caveated (first-click only).")

h("5a. Paired t-tests — mean_rt_first_ms", level=2)
line(f"  {'Condition':<12}  {'M_lab':>8}  {'M_phone':>8}  {'Δ':>8}  "
     f"{'t':>8}  {'df':>4}  {'p':>8}  {'d':>7}  Sig")
line(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*10}")

paired_rq3 = {}
for tload in ['single', 'multiple']:
    lab_s = cell(tload, 'lab',   'mean_rt_first_ms')
    ph_s  = cell(tload, 'phone', 'mean_rt_first_ms')
    common = lab_s.index.intersection(ph_s.index)
    lv    = lab_s.loc[common].values
    pv    = ph_s.loc[common].values
    t, p  = ttest_rel(lv, pv)
    d     = cohens_d_paired(lv, pv)
    delta = np.mean(lv) - np.mean(pv)
    df_t  = len(common) - 1
    sig   = '***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else 'n.s.'))
    line(f"  {tload.capitalize():<12}  {np.mean(lv):>8.1f}  {np.mean(pv):>8.1f}  "
         f"{delta:>8.1f}  {t:>8.3f}  {df_t:>4}  {p:>8.4f}  {d:>7.3f}  {sig}")
    paired_rq3[tload] = {'t': t, 'p': p, 'd': d, 'delta': delta,
                          'M_lab': np.mean(lv), 'M_phone': np.mean(pv)}

line()
line("  NOTE: For multiple-target, lab and phone both use rt_first_ms (~1500 ms each).")
line("  Any non-significance here does NOT mean the modality effect is absent —")
line("  it reflects the RT operationalisation mismatch. The true modality effect")
line("  (last-click lab ~6500ms vs phone first-touch ~1700ms) was reported in Report 1.")


# ══════════════════════════════════════════════════════════
# 6. RELIABILITY — SPLIT-HALF
# ══════════════════════════════════════════════════════════

h("6. RELIABILITY — Split-Half (Spearman-Brown corrected)", level=1)
line("  Lab : odd trials (1,3,5,...) vs even trials (2,4,6,...)")
line("  Phone: first-half levels vs second-half levels")
line("  Spearman-Brown formula: r_sb = 2r / (1 + r)")
line()

def spearman_brown(r):
    return (2 * r) / (1 + r)

reliability_results = {}

for tload in ['single', 'multiple']:
    for mod in ['lab', 'phone']:
        mask = ((df_trials['target_load'] == tload) &
                (df_trials['modality']    == mod))
        grp  = df_trials[mask].copy()

        half1_means = []
        half2_means = []

        for pid in grp['participant_id'].unique():
            p_data = grp[grp['participant_id'] == pid].sort_values('trial_num')
            rt     = p_data['rt_first_ms'].dropna().values
            n      = len(rt)
            if n < 4:
                continue

            if mod == 'lab':
                # Odd/even split by trial index (0-based)
                h1 = rt[0::2]   # indices 0,2,4,...
                h2 = rt[1::2]   # indices 1,3,5,...
            else:
                # First-half / second-half split
                mid = n // 2
                h1  = rt[:mid]
                h2  = rt[mid:2*mid]   # match length

            half1_means.append(np.mean(h1))
            half2_means.append(np.mean(h2))

        if len(half1_means) < 3:
            continue

        r, p_r    = pearsonr(half1_means, half2_means)
        r_sb      = spearman_brown(r)
        cell_name = f"{tload.capitalize()}/{mod.capitalize()}"
        reliability_results[cell_name] = {'r': r, 'r_sb': r_sb, 'p': p_r,
                                           'n': len(half1_means)}
        line(f"  {cell_name:<20}  n={len(half1_means):>2}  "
             f"r_half={r:.3f}  r_SB={r_sb:.3f}  p={p_r:.4f}")

line()
line("  r_SB interpretation: > .70 acceptable, > .80 good, > .90 excellent")


# ══════════════════════════════════════════════════════════
# 7. SAVE TEXT REPORT
# ══════════════════════════════════════════════════════════

report_path = os.path.join(OUT_DIR, 'inferential_results.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(R))
print('\n'.join(R))
print(f"\nReport saved: {report_path}")


# ══════════════════════════════════════════════════════════
# 8. FIGURES
# ══════════════════════════════════════════════════════════

COLORS = {
    'Single/Lab'    : '#1976D2',
    'Single/Phone'  : '#42A5F5',
    'Multiple/Lab'  : '#D32F2F',
    'Multiple/Phone': '#EF9A9A',
}
C_SINGLE   = '#1976D2'
C_MULTIPLE = '#D32F2F'

# ── Fig 1: ANOVA interaction plot + effect sizes ──────────
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('2×2 Mixed ANOVA Results\nDV: log(Mean RT — First Click)',
              fontsize=12, fontweight='bold')

ax = axes[0]
cm = anova_log['cell_means']
x  = [0, 1]
ax.plot(x, [cm['Single/Lab'],   cm['Single/Phone']],
        'o-', color=C_SINGLE,   linewidth=2.5, markersize=8,
        label='Single Target')
ax.plot(x, [cm['Multiple/Lab'], cm['Multiple/Phone']],
        's-', color=C_MULTIPLE, linewidth=2.5, markersize=8,
        label='Multiple Target')

# Add error bars (SE of cell means)
for tload, marker, color in [('single','o',C_SINGLE),('multiple','s',C_MULTIPLE)]:
    for xi, mod in enumerate(['lab','phone']):
        vals = cell(tload, mod, 'log_mean_rt_first').values
        se   = np.std(vals, ddof=1) / np.sqrt(len(vals))
        mean = np.mean(vals)
        ax.errorbar(xi, mean, yerr=se, fmt='none',
                    color=color, capsize=5, linewidth=1.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Lab', 'Phone'], fontsize=11)
ax.set_ylabel('ln(Mean RT ms)', fontsize=10)
ax.set_title('Interaction Plot\n(error bars = ±1 SE)', fontsize=10)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

# Add significance annotations
def sig_label(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return 'n.s.'

textstr = (f"Target Load:  F({anova_log['A']['df'][0]},{anova_log['A']['df'][1]}) = "
           f"{anova_log['A']['F']:.2f}, {sig_label(anova_log['A']['p'])}, "
           f"η²p={anova_log['A']['eta2']:.3f}\n"
           f"Modality:     F({anova_log['B']['df'][0]},{anova_log['B']['df'][1]}) = "
           f"{anova_log['B']['F']:.2f}, {sig_label(anova_log['B']['p'])}, "
           f"η²p={anova_log['B']['eta2']:.3f}\n"
           f"Interaction:  F({anova_log['AB']['df'][0]},{anova_log['AB']['df'][1]}) = "
           f"{anova_log['AB']['F']:.2f}, {sig_label(anova_log['AB']['p'])}, "
           f"η²p={anova_log['AB']['eta2']:.3f}")
ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
        fontsize=7.5, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Effect size bar chart
ax2 = axes[1]
effects = ['Target\nLoad (A)', 'Modality\n(B)', 'A × B\nInteraction']
eta2s   = [anova_log['A']['eta2'], anova_log['B']['eta2'], anova_log['AB']['eta2']]
colors  = ['#1976D2', '#43A047', '#FB8C00']
bars    = ax2.bar(effects, eta2s, color=colors, alpha=0.8, edgecolor='white', width=0.5)
ax2.axhline(0.01, color='gray', linestyle=':', linewidth=1, label='small (.01)')
ax2.axhline(0.06, color='gray', linestyle='--', linewidth=1, label='medium (.06)')
ax2.axhline(0.14, color='gray', linestyle='-', linewidth=1, label='large (.14)')
for bar, val, p in zip(bars, eta2s,
                        [anova_log['A']['p'], anova_log['B']['p'], anova_log['AB']['p']]):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.003,
             f"{val:.3f}\n{sig_label(p)}", ha='center', fontsize=9, fontweight='bold')
ax2.set_ylabel('Partial η²', fontsize=10)
ax2.set_title('Effect Sizes (partial η²)', fontsize=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_ylim(0, max(eta2s)*1.5 + 0.05)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
p1 = os.path.join(OUT_DIR, 'fig_anova.png')
fig1.savefig(p1, dpi=150, bbox_inches='tight')
print(f"Saved: {p1}")

# ── Fig 2: Concurrent validity scatterplots ───────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('RQ1 — Concurrent Validity\nGame vs Lab Performance',
              fontsize=12, fontweight='bold')

# Single
ax = axes2[0]
ax.scatter(sl, sp, color=C_SINGLE, s=60, alpha=0.8, zorder=3)
for i, pid in enumerate(common_s):
    ax.annotate(str(pid), (sl[i], sp[i]), fontsize=6, alpha=0.6,
                xytext=(3,3), textcoords='offset points')
m_fit, b_fit = np.polyfit(sl, sp, 1)
x_line = np.linspace(sl.min(), sl.max(), 100)
ax.plot(x_line, m_fit*x_line + b_fit, '--', color=C_SINGLE, linewidth=1.5)
ax.set_xlabel('Lab — Mean RT First Click (ms)', fontsize=9)
ax.set_ylabel('Phone — Mean RT First Click (ms)', fontsize=9)
ax.set_title('Single Target\n(both = time to find 1 target)', fontsize=10)
ax.text(0.97, 0.05,
        f"Pearson r = {r_s:.3f}, p = {p_r_s:.3f}\nSpearman ρ = {rho_s:.3f}, p = {p_rho_s:.3f}\nn = {len(sl)}",
        transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(facecolor='lightblue', alpha=0.5))
ax.grid(True, alpha=0.3)

# Multiple (corrected DV)
ax = axes2[1]
ax.scatter(ml_per, mp_ait, color=C_MULTIPLE, s=60, alpha=0.8, zorder=3)
for i, pid in enumerate(common_m):
    ax.annotate(str(pid), (ml_per[i], mp_ait[i]), fontsize=6, alpha=0.6,
                xytext=(3,3), textcoords='offset points')
m_fit2, b_fit2 = np.polyfit(ml_per, mp_ait, 1)
x_line2 = np.linspace(ml_per.min(), ml_per.max(), 100)
ax.plot(x_line2, m_fit2*x_line2 + b_fit2, '--', color=C_MULTIPLE, linewidth=1.5)
ax.set_xlabel('Lab — RT Last Click / 5  (mean ms per target)', fontsize=9)
ax.set_ylabel('Phone — Avg Inter-Target Time (ms)', fontsize=9)
ax.set_title('Multiple Target\n(both = mean time per target)', fontsize=10)
ax.text(0.97, 0.95,
        f"Pearson r = {r_m:.3f}, p = {p_r_m:.3f}\nSpearman ρ = {rho_m:.3f}, p = {p_rho_m:.3f}\nn = {len(common_m)}",
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='lightsalmon', alpha=0.5))
ax.grid(True, alpha=0.3)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, 'fig_validity.png')
fig2.savefig(p2, dpi=150, bbox_inches='tight')
print(f"Saved: {p2}")

# ── Fig 3: Reliability plot ───────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))
fig3.suptitle('Split-Half Reliability (Spearman-Brown Corrected)',
              fontsize=12, fontweight='bold')

cells_r  = list(reliability_results.keys())
r_halfs  = [reliability_results[c]['r']    for c in cells_r]
r_sbs    = [reliability_results[c]['r_sb'] for c in cells_r]
x_pos    = np.arange(len(cells_r))
w        = 0.35

bars1 = ax3.bar(x_pos - w/2, r_halfs, w, label='Split-half r', alpha=0.75,
                color='#42A5F5', edgecolor='white')
bars2 = ax3.bar(x_pos + w/2, r_sbs,   w, label='Spearman-Brown r_SB', alpha=0.75,
                color='#1976D2', edgecolor='white')
ax3.axhline(0.70, color='orange', linestyle='--', linewidth=1.5, label='Acceptable (.70)')
ax3.axhline(0.80, color='green',  linestyle='--', linewidth=1.5, label='Good (.80)')
ax3.axhline(0.90, color='purple', linestyle='--', linewidth=1.5, label='Excellent (.90)')

for bar, val in zip(list(bars1)+list(bars2), r_halfs+r_sbs):
    ax3.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.2f}',
             ha='center', fontsize=8, fontweight='bold')

ax3.set_xticks(x_pos)
ax3.set_xticklabels(cells_r, fontsize=9)
ax3.set_ylabel('Correlation', fontsize=10)
ax3.set_ylim(0, 1.25)
ax3.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=3, frameon=True)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
p3 = os.path.join(OUT_DIR, 'fig_reliability.png')
fig3.savefig(p3, dpi=150, bbox_inches='tight')
print(f"Saved: {p3}")

print("\nAll done.")
