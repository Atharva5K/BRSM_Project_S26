import os, ast, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)


BASE       = os.path.join('Attention Task Validation', 'data_brsm')
OUTPUT_DIR = '.'         
ALPHA      = 0.05

print("=" * 65)
print("  BRSM Project — Descriptive Statistics  (Full Dataset)")
print("=" * 65)
print(f"  Working dir : {os.getcwd()}")
print(f"  Data root   : {os.path.abspath(BASE)}")
print()

for sub in ['single/lab', 'single/phone', 'multiple/lab', 'multiple/phone']:
    path   = os.path.join(BASE, sub)
    exists = os.path.isdir(path)
    n      = len([f for f in os.listdir(path) if f.endswith('.csv')]) if exists else 0
    print(f"  {'✓' if exists else '✗'}  {sub:<25}  {n:>2} CSV files")
print()

def _parse_rt(val, take='last'):
    if pd.isna(val):
        return np.nan
    try:
        p = ast.literal_eval(str(val))
        if isinstance(p, list) and p:
            return float(p[-1] if take == 'last' else p[0])
        return float(p)
    except:
        return np.nan

def _count_unique_targets(clicked_str):
    if pd.isna(clicked_str):
        return 0
    try:
        p = ast.literal_eval(str(clicked_str))
        if isinstance(p, list):
            return len(set(x for x in p if 'target' in str(x).lower()))
        return 1 if 'target' in str(clicked_str).lower() else 0
    except:
        return 1 if 'target' in str(clicked_str).lower() else 0


def load_lab(filepath, pid, target_load):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"    [ERROR] {filepath}: {e}")
        return pd.DataFrame()

    trials = df[df['target_col'].notna()].copy()
    if len(trials) == 0:
        return pd.DataFrame()

    rt_col  = ('trials.mouse.time'
               if 'trials.mouse.time' in trials.columns
               else 'mouse.time')
    clk_col = ('trials.mouse.clicked_name'
               if 'trials.mouse.clicked_name' in trials.columns
               else 'mouse.clicked_name')

    rt_last   = trials[rt_col].apply(lambda x: _parse_rt(x, 'last'))  * 1000
    rt_first  = trials[rt_col].apply(lambda x: _parse_rt(x, 'first')) * 1000
    n_correct = trials[clk_col].apply(_count_unique_targets)

    n_req    = 1 if target_load == 'single' else 5
    accuracy = (n_correct / n_req).clip(0, 1)

    return pd.DataFrame({
        'participant':        int(pid),
        'target_load':        target_load,
        'modality':           'lab',
        'level':              trials['trials.thisN'].values + 1,
        'trial_num':          trials['trials.thisN'].values,
        'target_col':         trials['target_col'].values,
        'rt_ms':              rt_last.values,
        'rt_first_ms':        rt_first.values,
        'accuracy':           accuracy.values,
        'correct':            (n_correct >= 1).astype(int).values,
        'n_correct':          n_correct.values,
        'false_alarms':       np.zeros(len(trials), dtype=int),
        'hit_rate':           accuracy.values,
        'final_score':        np.full(len(trials), np.nan),
        'avg_inter_target_ms': np.full(len(trials), np.nan),
    })


def load_game(filepath, pid, target_load):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"    [ERROR] {filepath}: {e}")
        return pd.DataFrame()

    if len(df) == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        'participant':        int(pid),
        'target_load':        target_load,
        'modality':           'game',
        'level':              df['Level'].astype(int).values,
        'trial_num':          (df['Level'] - 1).astype(int).values,
        'target_col':         'NA',
        'rt_ms':              df['InitialResponseTime(ms)'].astype(float).values,
        'rt_first_ms':        df['InitialResponseTime(ms)'].astype(float).values,
        'accuracy':           (df['SuccessRate(%)'] / 100).values,
        'correct':            df['Completed'].astype(int).values,
        'n_correct':          np.zeros(len(df), dtype=int),
        'false_alarms':       df['FalseAlarms'].astype(int).values,
        'hit_rate':           (df['HitRate(%)'] / 100).values,
        'final_score':        df['FinalScore'].astype(int).values,
        'avg_inter_target_ms': df['AvgInterTargetTime(ms)'].astype(float).values,
    })


def load_all(base_dir):
    cfg = [
        ('single',   'lab',   load_lab,  range(1,  22)),
        ('single',   'phone', load_game, range(1,  22)),
        ('multiple', 'lab',   load_lab,  range(22, 38)),
        ('multiple', 'phone', load_game, range(22, 38)),
    ]
    all_dfs = []
    for condition, folder, loader_fn, _ in cfg:
        folder_path = os.path.join(base_dir, condition, folder)
        if not os.path.isdir(folder_path):
            print(f"  [SKIP] not found: {folder_path}")
            continue
        csv_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.csv'))
        print(f"  Loading {condition}/{folder} ({len(csv_files)} files)…", end=' ')
        loaded = 0
        for fname in csv_files:
            pid = int(fname.split('_')[0])
            df  = loader_fn(os.path.join(folder_path, fname), pid, condition)
            if len(df) > 0:
                all_dfs.append(df)
                loaded += 1
        print(f"{loaded} OK")

    print()
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

data = load_all(BASE)

if len(data) == 0:
    raise SystemExit("No data loaded. Check BASE path and re-run.")

n_ppt   = data['participant'].nunique()
n_rows  = len(data)
print(f"  Loaded {n_rows} rows from {n_ppt} participants")
print(f"  Conditions : {sorted(data['target_load'].unique())}")
print(f"  Modalities : {sorted(data['modality'].unique())}")
print(f"  Level range: {int(data['level'].min())} – {int(data['level'].max())}\n")

def section(title):
    print("\n" + "═" * 65)
    print(f"  {title}")
    print("═" * 65)

def sub(title):
    print(f"\n  ── {title}")

def ci95(series):
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    return stats.t.ppf(0.975, df=len(s) - 1) * s.sem()

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pool = np.sqrt(((na-1)*a.std()**2 + (nb-1)*b.std()**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pool if pool > 0 else np.nan

def interp_d(d):
    if pd.isna(d): return "n/a"
    d = abs(d)
    if d < 0.2:  return "trivial"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"

ppt_means = (data
    .groupby(['participant', 'target_load', 'modality'])
    .agg(mean_rt  = ('rt_ms',        'mean'),
         mean_acc = ('accuracy',     'mean'),
         sum_fa   = ('false_alarms', 'sum'))
    .reset_index())

section("A. GENERAL DESCRIPTIVE STATISTICS")

sub("A1. Per-Participant Summary")
ppt_full = (data
    .groupby(['participant', 'target_load', 'modality'])
    .agg(N         = ('rt_ms',        'count'),
         M_RT      = ('rt_ms',        'mean'),
         SD_RT     = ('rt_ms',        'std'),
         Median_RT = ('rt_ms',        'median'),
         IQR_RT    = ('rt_ms',        lambda x: x.quantile(.75) - x.quantile(.25)),
         Min_RT    = ('rt_ms',        'min'),
         Max_RT    = ('rt_ms',        'max'),
         M_Acc     = ('accuracy',     'mean'),
         SD_Acc    = ('accuracy',     'std'),
         Total_FA  = ('false_alarms', 'sum'))
    .round(2))
print(ppt_full.to_string())

sub("A2. 2×2 Design Cell Summary  [target_load × modality]")
cell = (ppt_means
    .groupby(['target_load', 'modality'])
    .agg(N_ppt     = ('participant', 'count'),
         M_RT      = ('mean_rt',  'mean'),
         SD_RT     = ('mean_rt',  'std'),
         SE_RT     = ('mean_rt',  'sem'),
         Median_RT = ('mean_rt',  'median'),
         Min_RT    = ('mean_rt',  'min'),
         Max_RT    = ('mean_rt',  'max'),
         M_Acc     = ('mean_acc', 'mean'),
         SD_Acc    = ('mean_acc', 'std'),
         M_FA      = ('sum_fa',   'mean'),
         SD_FA     = ('sum_fa',   'std'))
    .round(3))
print(cell.to_string())

sub("A3. 95% Confidence Intervals on Mean RT")
print(f"\n  {'Condition':<12} {'Modality':<8} {'N':>4}  "
      f"{'M_RT':>9}  {'95%_CI':>18}")
print("  " + "-" * 56)
for (load, mod), g in ppt_means.groupby(['target_load', 'modality']):
    m  = g['mean_rt'].mean()
    hw = ci95(g['mean_rt'])
    print(f"  {load:<12} {mod:<8} {len(g):>4}  "
          f"{m:>9.1f}  [{m-hw:>8.1f}, {m+hw:>8.1f}]")

sub("A4. Normality Tests (Shapiro-Wilk) — RT per participant × modality")
norm_rows = []
for (pid, load, mod), g in data.groupby(['participant', 'target_load', 'modality']):
    rt = g['rt_ms'].dropna()
    if len(rt) < 3:
        continue
    sw, p_sw = stats.shapiro(rt)
    norm_rows.append(dict(
        PID=pid, Condition=load, Modality=mod, N=len(rt),
        Skew=round(stats.skew(rt), 3),
        Kurt=round(stats.kurtosis(rt), 3),
        SW_W=round(sw, 4), SW_p=round(p_sw, 4),
        Normal='Yes' if p_sw >= ALPHA else 'NO'))
norm_df = pd.DataFrame(norm_rows)
print(norm_df.to_string(index=False))

n_norm = (norm_df['Normal'] == 'Yes').sum()
print(f"\n  Summary: {n_norm}/{len(norm_df)} distributions are approximately normal")
print("  → Use log-transformed RT or non-parametric tests where normality fails")

sub("A5. Overall RT Stats per Cell (all trials pooled)")
for (load, mod), g in data.groupby(['target_load', 'modality']):
    rt = g['rt_ms'].dropna()
    print(f"\n  {load.upper()} / {mod.upper()}"
          f"  ({g['participant'].nunique()} ppts, {len(rt)} trials)")
    print(f"    Mean ± SD    : {rt.mean():.1f} ± {rt.std():.1f} ms")
    print(f"    Median [IQR] : {rt.median():.1f}  "
          f"[{rt.quantile(.25):.1f} – {rt.quantile(.75):.1f}]")
    print(f"    Range        : {rt.min():.1f} – {rt.max():.1f} ms")
    print(f"    Skewness     : {stats.skew(rt):.3f}   "
          f"Kurtosis: {stats.kurtosis(rt):.3f}")

sub("A6. Data Quality Audit")
print(f"\n  {'Metric':<38} {'Lab':>7}  {'Game':>7}  {'Total':>7}")
print("  " + "-" * 60)
for label, fn in [
    ("Total trials / levels",    lambda g: len(g)),
    ("Missing RT",               lambda g: int(g['rt_ms'].isna().sum())),
    ("Incomplete (correct = 0)", lambda g: int((g['correct'] == 0).sum())),
    ("Outliers (±3 SD in RT)",   lambda g: int(
        (np.abs(g['rt_ms'] - g['rt_ms'].mean()) > 3 * g['rt_ms'].std()).sum())),
]:
    vl = fn(data[data['modality'] == 'lab'])
    vg = fn(data[data['modality'] == 'game'])
    print(f"  {label:<38} {vl:>7}  {vg:>7}  {vl+vg:>7}")

sub("A7. False Alarms Summary  (game modality only)")
fa = (data[data['modality'] == 'game']
      .groupby(['participant', 'target_load'])['false_alarms'].sum()
      .reset_index())
fa_cell = (fa.groupby('target_load')['false_alarms']
             .agg(N='count', M='mean', SD='std',
                  Median='median', Min='min', Max='max')
             .round(3))
print(fa_cell.to_string())

section("B. RQ1 — CONCURRENT VALIDITY  (Game vs Lab Correlation)")
print("""
  H0: No correlation between game and lab RT / accuracy.
  H1: Significant positive correlation — game is a valid substitute.
  Method: Pearson r and Spearman ρ on per-participant mean RT,
          computed separately for single and multiple conditions.
""")

sub("B1. Per-participant mean RT pivot  (lab vs game)")
for load in ['single', 'multiple']:
    pivot = (ppt_means[ppt_means['target_load'] == load]
             .pivot(index='participant', columns='modality', values='mean_rt')
             .dropna())
    print(f"\n  {load.upper()}  (n = {len(pivot)} matched participants)")
    print(pivot.round(1).to_string())

    if len(pivot) >= 3 and 'lab' in pivot.columns and 'game' in pivot.columns:
        r_p,  p_p  = stats.pearsonr( pivot['lab'], pivot['game'])
        r_sp, p_sp = stats.spearmanr(pivot['lab'], pivot['game'])
        print(f"\n    Pearson  r   = {r_p:.3f},  p = {p_p:.4f}  "
              f"({'sig.' if p_p < ALPHA else 'n.s.'})")
        print(f"    Spearman rho = {r_sp:.3f}, p = {p_sp:.4f}  "
              f"({'sig.' if p_sp < ALPHA else 'n.s.'})")
        direction = "positive ✓" if r_p > 0 else "negative ✗"
        print(f"    Direction: {direction}")
    else:
        print("    (Need ≥ 3 matched participants to compute correlation)")

sub("B2. Validity — Accuracy")
for load in ['single', 'multiple']:
    pivot = (ppt_means[ppt_means['target_load'] == load]
             .pivot(index='participant', columns='modality', values='mean_acc')
             .dropna())
    if len(pivot) >= 3 and 'lab' in pivot.columns and 'game' in pivot.columns:
        r_p, p_p = stats.pearsonr(pivot['lab'], pivot['game'])
        print(f"  {load.upper()}: Pearson r = {r_p:.3f},  p = {p_p:.4f}  "
              f"({'sig.' if p_p < ALPHA else 'n.s.'})")

section("C. RQ2 — SINGLE vs MULTIPLE TARGET CONDITION")
print("""
  H0: No RT / accuracy difference between conditions.
  H1: Multiple target → higher RT, lower accuracy.
  Test: Independent-samples Welch t-test on per-participant means.
""")

sub("C1. RT — Single vs Multiple")
for mod in ['lab', 'game']:
    g = ppt_means[ppt_means['modality'] == mod]
    s = g[g['target_load'] == 'single'  ]['mean_rt'].dropna()
    m = g[g['target_load'] == 'multiple']['mean_rt'].dropna()
    if len(s) < 2 or len(m) < 2:
        continue
    t_val, p_val = stats.ttest_ind(s, m, equal_var=False)
    d            = cohens_d(m, s)
    print(f"\n  {mod.upper()}  (n_single={len(s)}, n_multi={len(m)}):")
    print(f"    Single   : M = {s.mean():>8.1f} ms,  SD = {s.std():.1f},  "
          f"95%CI [{s.mean()-ci95(s):.1f}, {s.mean()+ci95(s):.1f}]")
    print(f"    Multiple : M = {m.mean():>8.1f} ms,  SD = {m.std():.1f},  "
          f"95%CI [{m.mean()-ci95(m):.1f}, {m.mean()+ci95(m):.1f}]")
    print(f"    Difference: {m.mean()-s.mean():+.1f} ms")
    print(f"    Welch t = {t_val:.3f},  p = {p_val:.4f}  "
          f"({'sig.' if p_val < ALPHA else 'n.s.'} at α={ALPHA})")
    print(f"    Cohen's d = {d:.3f}  ({interp_d(d)} effect)")

sub("C2. Accuracy — Single vs Multiple")
for mod in ['lab', 'game']:
    g = ppt_means[ppt_means['modality'] == mod]
    s = g[g['target_load'] == 'single'  ]['mean_acc'].dropna()
    m = g[g['target_load'] == 'multiple']['mean_acc'].dropna()
    if len(s) < 2 or len(m) < 2:
        continue
    t_val, p_val = stats.ttest_ind(s, m, equal_var=False)
    d            = cohens_d(s, m)
    print(f"\n  {mod.upper()}:  Single M={s.mean():.3f} SD={s.std():.3f}  |  "
          f"Multiple M={m.mean():.3f} SD={m.std():.3f}")
    print(f"    t = {t_val:.3f},  p = {p_val:.4f},  d = {d:.3f} ({interp_d(d)})")

sub("C3. False Alarms — Single vs Multiple  (game only)")
fa_s = data[(data['modality']=='game') & (data['target_load']=='single')  ]['false_alarms']
fa_m = data[(data['modality']=='game') & (data['target_load']=='multiple')]['false_alarms']
print(f"  Single   : M={fa_s.mean():.2f},  SD={fa_s.std():.2f},  Total={int(fa_s.sum())}")
print(f"  Multiple : M={fa_m.mean():.2f},  SD={fa_m.std():.2f},  Total={int(fa_m.sum())}")

section("D. RQ3 — MODALITY EFFECT  (Lab vs Game)")
print("""
  H0: No RT / accuracy difference between modalities.
  H1: Game interface alters performance relative to lab.
  Test: Paired t-test on per-participant means (within-subjects).
        Full 2×2 Mixed ANOVA → Report 2.
""")

sub("D1. RT — Paired Lab vs Game  (per condition)")
for load in ['single', 'multiple']:
    pivot = (ppt_means[ppt_means['target_load'] == load]
             .pivot(index='participant', columns='modality', values='mean_rt')
             .dropna())
    if 'lab' not in pivot.columns or 'game' not in pivot.columns:
        print(f"  {load.upper()}: both modalities not available — check data")
        continue
    diff         = pivot['lab'] - pivot['game']
    t_val, p_val = stats.ttest_rel(pivot['lab'], pivot['game'])
    d            = diff.mean() / diff.std() if diff.std() > 0 else np.nan
    print(f"\n  {load.upper()}  (n = {len(pivot)} matched participants):")
    print(f"    Lab  : M = {pivot['lab'].mean():>8.1f} ms,  SD = {pivot['lab'].std():.1f}")
    print(f"    Game : M = {pivot['game'].mean():>8.1f} ms,  SD = {pivot['game'].std():.1f}")
    print(f"    Mean diff (Lab − Game) : {diff.mean():+.1f} ms")
    print(f"    Paired t({len(pivot)-1}) = {t_val:.3f},  p = {p_val:.4f}  "
          f"({'sig.' if p_val < ALPHA else 'n.s.'})")
    print(f"    Cohen's d = {d:.3f}  ({interp_d(d)})")

sub("D2. Accuracy — Paired Lab vs Game")
for load in ['single', 'multiple']:
    pivot = (ppt_means[ppt_means['target_load'] == load]
             .pivot(index='participant', columns='modality', values='mean_acc')
             .dropna())
    if 'lab' not in pivot.columns or 'game' not in pivot.columns:
        continue
    diff         = pivot['lab'] - pivot['game']
    t_val, p_val = stats.ttest_rel(pivot['lab'], pivot['game'])
    print(f"  {load.upper()}: Lab M={pivot['lab'].mean():.3f}  "
          f"Game M={pivot['game'].mean():.3f}  "
          f"Diff={diff.mean():+.3f}  "
          f"t={t_val:.3f}  p={p_val:.4f}  "
          f"({'sig.' if p_val < ALPHA else 'n.s.'})")

section("E. RQ4 — PRACTICE / LEVEL EFFECT")
print("""
  H0: RT does not change across levels / trials.
  H1: RT decreases (practice) or increases (difficulty) across levels.
  Test: Pearson r and Spearman ρ  (level number vs RT).
""")

sub("E1. Level vs RT correlation  (per participant × modality)")
prac_rows = []
for (pid, load, mod), g in data.groupby(['participant', 'target_load', 'modality']):
    d = g.sort_values('level')[['level', 'rt_ms']].dropna()
    if len(d) < 3:
        continue
    r,   p_r   = stats.pearsonr( d['level'], d['rt_ms'])
    rho, p_rho = stats.spearmanr(d['level'], d['rt_ms'])
    prac_rows.append(dict(
        PID=pid, Condition=load, Modality=mod, N=len(d),
        Pearson_r=round(r, 3),   p_pearson=round(p_r, 4),
        Spearman_rho=round(rho, 3), p_spearman=round(p_rho, 4),
        Trend='↓ faster' if r < 0 else '↑ slower',
        Sig='*' if p_r < ALPHA else ''))
prac_df = pd.DataFrame(prac_rows)
print(prac_df.to_string(index=False))

n_sig = (prac_df['Sig'] == '*').sum()
print(f"\n  {n_sig} / {len(prac_df)} participants show a significant linear trend "
      f"(α = {ALPHA})")

sub("E2. Group mean RT by level  (per condition × modality)")
level_summary = (data
    .groupby(['target_load', 'modality', 'level'])['rt_ms']
    .agg(N='count', M_RT='mean', SD_RT='std', SE_RT='sem')
    .reset_index()
    .round(2))
print(level_summary.to_string(index=False))

section("F. GENERATING FIGURES")

COND_COL  = {'single': '#1976D2', 'multiple': '#D32F2F'}
MOD_HATCH = {'lab': '///', 'game': ''}

fig = plt.figure(figsize=(18, 20))
fig.suptitle(
    f'Descriptive Statistics — Attention Task Validation\n'
    f'N = {n_ppt} participants  (Single n=21, Multiple n=16)',
    fontsize=14, fontweight='bold', y=0.995)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.40)

ax1 = fig.add_subplot(gs[0, :2])
order = [('single','lab'), ('single','game'), ('multiple','lab'), ('multiple','game')]
box_data, xlabels, bcolors, bhatch = [], [], [], []
for (load, mod) in order:
    rt = data[(data['target_load']==load) & (data['modality']==mod)]['rt_ms'].dropna()
    box_data.append(rt.values)
    xlabels.append(f"{load[:3].upper()}\n{mod}")
    bcolors.append(COND_COL[load])
    bhatch.append(MOD_HATCH[mod])
bp = ax1.boxplot(box_data, patch_artist=True,
                 medianprops=dict(color='black', linewidth=2),
                 flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, col, h in zip(bp['boxes'], bcolors, bhatch):
    patch.set_facecolor(col); patch.set_alpha(0.72); patch.set_hatch(h)
ax1.set_xticklabels(xlabels, fontsize=9)
ax1.set_ylabel('RT (ms)', fontsize=10)
ax1.set_title('RT Distribution — All 4 Design Cells', fontweight='bold')
ax1.legend(handles=[
    Patch(facecolor=COND_COL['single'],  label='Single target'),
    Patch(facecolor=COND_COL['multiple'],label='Multiple targets'),
    Patch(facecolor='grey', hatch='///', label='Lab task'),
    Patch(facecolor='grey', hatch='',   label='Game'),
], fontsize=8, loc='upper right')

ax2 = fig.add_subplot(gs[0, 2])
x, w = np.arange(2), 0.35
for i, mod in enumerate(['lab', 'game']):
    means, errs = [], []
    for load in ['single', 'multiple']:
        g = ppt_means[(ppt_means['target_load']==load) & (ppt_means['modality']==mod)]['mean_rt']
        means.append(g.mean())
        errs.append(ci95(g))
    ax2.bar(x + i*w, means, w, yerr=errs, capsize=5,
            label=mod.capitalize(), hatch=MOD_HATCH[mod],
            color=[COND_COL['single'], COND_COL['multiple']],
            alpha=0.82, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x + w/2); ax2.set_xticklabels(['Single', 'Multiple'])
ax2.set_ylabel('Mean RT (ms)')
ax2.set_title('Mean RT ± 95% CI\n(per-participant means)', fontweight='bold')
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[1, 0])
for i, mod in enumerate(['lab', 'game']):
    means = []
    for load in ['single', 'multiple']:
        g = ppt_means[(ppt_means['target_load']==load) & (ppt_means['modality']==mod)]['mean_acc']
        means.append(g.mean())
    ax3.bar(x + i*w, means, w, label=mod.capitalize(),
            hatch=MOD_HATCH[mod],
            color=[COND_COL['single'], COND_COL['multiple']],
            alpha=0.82, edgecolor='black', linewidth=0.5)
ax3.set_xticks(x + w/2); ax3.set_xticklabels(['Single', 'Multiple'])
ax3.set_ylim(0, 1.18)
ax3.set_ylabel('Mean Accuracy (proportion)')
ax3.set_title('Mean Accuracy by Condition', fontweight='bold')
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[1, 1])
for load in ['single', 'multiple']:
    pivot = (ppt_means[ppt_means['target_load']==load]
             .pivot(index='participant', columns='modality', values='mean_rt')
             .dropna())
    if 'lab' in pivot.columns and 'game' in pivot.columns and len(pivot) >= 2:
        ax4.scatter(pivot['lab'], pivot['game'],
                    color=COND_COL[load], label=load.capitalize(),
                    s=50, alpha=0.8, edgecolors='white', linewidths=0.5)
        if len(pivot) >= 3:
            m_fit, b_fit = np.polyfit(pivot['lab'], pivot['game'], 1)
            xr = np.linspace(pivot['lab'].min(), pivot['lab'].max(), 100)
            ax4.plot(xr, m_fit*xr + b_fit, '--',
                     color=COND_COL[load], linewidth=1.2)
ax4.set_xlabel('Lab Mean RT (ms)')
ax4.set_ylabel('Game Mean RT (ms)')
ax4.set_title('Lab vs Game RT\n(Concurrent Validity — RQ1)', fontweight='bold')
ax4.legend(fontsize=8)

ax5 = fig.add_subplot(gs[1, 2])
gm = data[(data['modality']=='game') & (data['target_load']=='multiple')]
if len(gm) > 0:
    fa_lv = gm.groupby('level')['false_alarms'].agg(['mean', 'sem'])
    ax5.bar(fa_lv.index, fa_lv['mean'],
            color=COND_COL['multiple'], alpha=0.8,
            edgecolor='black', linewidth=0.5,
            yerr=fa_lv['sem'], capsize=4)
    ax5.set_xlabel('Level')
    ax5.set_ylabel('Mean False Alarms ± SE')
    ax5.set_title('False Alarms by Level\n(Game, Multiple Target)', fontweight='bold')

for col_idx, mod in enumerate(['game', 'lab']):
    ax = fig.add_subplot(gs[2, col_idx])
    for load in ['single', 'multiple']:
        grp     = data[(data['modality']==mod) & (data['target_load']==load)]
        lv_mean = grp.groupby('level')['rt_ms'].mean()
        lv_sem  = grp.groupby('level')['rt_ms'].sem()
        ax.plot(lv_mean.index, lv_mean.values,
                'o-', color=COND_COL[load], linewidth=1.8,
                markersize=4, label=load.capitalize())
        ax.fill_between(lv_mean.index,
                        lv_mean - lv_sem, lv_mean + lv_sem,
                        color=COND_COL[load], alpha=0.15)
    ax.set_xlabel('Level' if mod == 'game' else 'Trial Number')
    ax.set_ylabel('Mean RT (ms)')
    ax.set_title(f'Practice Effect — {mod.capitalize()}\n(mean ± SE)',
                 fontweight='bold')
    ax.legend(fontsize=8)

for idx, (load, mod) in enumerate(order[:3]):
    ax = fig.add_subplot(gs[3, idx])
    rt = data[(data['target_load']==load) & (data['modality']==mod)]['rt_ms'].dropna()
    if len(rt) < 5:
        continue
    sample    = rt.sample(min(len(rt), 50), random_state=42)
    sw, p_sw  = stats.shapiro(sample)
    (osm, osr), (slope, intercept, _) = stats.probplot(rt, dist='norm')
    ax.plot(osm, osr, 'o', color=COND_COL[load], markersize=3, alpha=0.55)
    ax.plot(osm, slope * np.array(osm) + intercept, '--k', linewidth=1)
    ax.set_title(f'Q-Q: {load[:3].upper()} / {mod}\nSW p = {p_sw:.3f}',
                 fontweight='bold', fontsize=9)
    ax.set_xlabel('Theoretical Quantiles', fontsize=8)
    ax.set_ylabel('Sample Quantiles', fontsize=8)
out_fig = os.path.join(OUTPUT_DIR, 'descriptive_stats_plots.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"  Figure saved → {out_fig}")

section("G. REPORT-READY SUMMARY TABLE")

print("\n  ── Cell-level summary (paste into Report 1 Results section)\n")
print(f"  {'Condition':<12} {'Modality':<8} {'N':>4}  "
      f"{'M_RT':>9}  {'SD_RT':>8}  {'95%CI':>20}  "
      f"{'M_Acc':>7}  {'M_FA':>6}")
print("  " + "-" * 78)
for (load, mod), g in ppt_means.groupby(['target_load', 'modality']):
    n   = len(g)
    m   = g['mean_rt'].mean()
    sd  = g['mean_rt'].std()
    hw  = ci95(g['mean_rt'])
    acc = g['mean_acc'].mean()
    fa  = g['sum_fa'].mean()
    print(f"  {load:<12} {mod:<8} {n:>4}  "
          f"{m:>9.1f}  {sd:>8.1f}  [{m-hw:>8.1f}, {m+hw:>8.1f}]  "
          f"{acc:>7.3f}  {fa:>6.2f}")

out_csv = os.path.join(OUTPUT_DIR, 'descriptive_stats_summary.csv')
ppt_means.to_csv(out_csv, index=False)
print(f"\n  Per-participant CSV saved → {out_csv}")

print("\n" + "=" * 65)
print("  All done!")
print("=" * 65)