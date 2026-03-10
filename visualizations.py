import os, ast, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

BASE       = os.path.join('Attention Task Validation', 'data_brsm')
OUTPUT_DIR = '.'

C_SINGLE   = '#2563EB'
C_MULTI    = '#DC2626'
C_LAB      = '#1E3A5F'
C_GAME     = '#0EA5E9'
C_LAB_M    = '#7F1D1D'
C_GAME_M   = '#FB923C'
GREY       = '#6B7280'
LIGHT_GREY = '#F3F4F6'

FONT_TITLE  = dict(fontsize=15, fontweight='bold', color='#111827')
FONT_LABEL  = dict(fontsize=11, color='#374151')
FONT_TICK   = dict(fontsize=10, color='#4B5563')
FONT_ANNOT  = dict(fontsize=9,  color='#374151')

def style_ax(ax, title='', xlabel='', ylabel='', legend=True):
    ax.set_facecolor(LIGHT_GREY)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#D1D5DB')
    ax.tick_params(colors='#4B5563', length=3)
    ax.grid(axis='y', color='white', linewidth=1.2, zorder=0)
    if title:   ax.set_title(title,   **FONT_TITLE, pad=10)
    if xlabel:  ax.set_xlabel(xlabel, **FONT_LABEL)
    if ylabel:  ax.set_ylabel(ylabel, **FONT_LABEL)
    ax.tick_params(labelsize=10)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved → {path}")
    plt.close(fig)

def _parse_rt(val, take='last'):
    if pd.isna(val): return np.nan
    try:
        p = ast.literal_eval(str(val))
        if isinstance(p, list) and p:
            return float(p[-1] if take == 'last' else p[0])
        return float(p)
    except: return np.nan

def _count_targets(s):
    if pd.isna(s): return 0
    try:
        p = ast.literal_eval(str(s))
        if isinstance(p, list):
            return len(set(x for x in p if 'target' in str(x).lower()))
        return 1 if 'target' in str(s).lower() else 0
    except: return 1 if 'target' in str(s).lower() else 0

def load_lab(fp, pid, load):
    try: df = pd.read_csv(fp)
    except: return pd.DataFrame()
    t = df[df['target_col'].notna()].copy()
    if len(t) == 0: return pd.DataFrame()
    rt_col  = 'trials.mouse.time' if 'trials.mouse.time' in t.columns else 'mouse.time'
    clk_col = 'trials.mouse.clicked_name' if 'trials.mouse.clicked_name' in t.columns else 'mouse.clicked_name'
    rt_last  = t[rt_col].apply(lambda x: _parse_rt(x,'last'))  * 1000
    rt_first = t[rt_col].apply(lambda x: _parse_rt(x,'first')) * 1000
    nc = t[clk_col].apply(_count_targets)
    nr = 1 if load == 'single' else 5
    return pd.DataFrame({
        'participant': int(pid), 'target_load': load, 'modality': 'lab',
        'level': t['trials.thisN'].values + 1,
        'rt_ms': rt_last.values, 'rt_first_ms': rt_first.values,
        'accuracy': (nc/nr).clip(0,1).values, 'correct': (nc>=1).astype(int).values,
        'false_alarms': np.zeros(len(t), int),
        'avg_inter_target_ms': np.full(len(t), np.nan),
    })

def load_game(fp, pid, load):
    try: df = pd.read_csv(fp)
    except: return pd.DataFrame()
    if len(df) == 0: return pd.DataFrame()
    return pd.DataFrame({
        'participant': int(pid), 'target_load': load, 'modality': 'game',
        'level': df['Level'].astype(int).values,
        'rt_ms': df['InitialResponseTime(ms)'].astype(float).values,
        'rt_first_ms': df['InitialResponseTime(ms)'].astype(float).values,
        'accuracy': (df['SuccessRate(%)']/100).values,
        'correct': df['Completed'].astype(int).values,
        'false_alarms': df['FalseAlarms'].astype(int).values,
        'avg_inter_target_ms': df['AvgInterTargetTime(ms)'].astype(float).values,
    })

def load_all(base_dir):
    cfg = [('single','lab',load_lab),('single','phone',load_game),
           ('multiple','lab',load_lab),('multiple','phone',load_game)]
    dfs = []
    for cond, folder, fn in cfg:
        fp = os.path.join(base_dir, cond, folder)
        if not os.path.isdir(fp): continue
        for fname in sorted(f for f in os.listdir(fp) if f.endswith('.csv')):
            pid = int(fname.split('_')[0])
            d   = fn(os.path.join(fp, fname), pid, cond)
            if len(d): dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

print("Loading data…")
data = load_all(BASE)
ppt = (data.groupby(['participant','target_load','modality'])
       .agg(mean_rt=('rt_ms','mean'), mean_acc=('accuracy','mean'),
            sum_fa=('false_alarms','sum'))
       .reset_index())
print(f"  {len(data)} rows, {data['participant'].nunique()} participants\n")

def ci95(s):
    s = s.dropna()
    if len(s)<2: return np.nan
    return stats.t.ppf(0.975, df=len(s)-1) * s.sem()

print("Fig 0 — Study overview…")
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('#F9FAFB')
ax.set_facecolor('#F9FAFB')
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('Study Design Overview — Attention Task Validation',
             fontsize=16, fontweight='bold', color='#111827', pad=14)

boxes = [
    (1.0, 3.5, C_SINGLE,  'SINGLE TARGET\nn = 21', 2.0, 1.2),
    (1.0, 1.5, C_MULTI,   'MULTIPLE TARGET\nn = 16', 2.0, 1.2),
    (4.5, 4.2, C_LAB,     'Lab Task\n(Visual Search)', 2.2, 0.9),
    (4.5, 2.8, C_GAME,    'Game\n(Attentional Spotter)', 2.2, 0.9),
    (4.5, 1.8, C_LAB_M,   'Lab Task\n(Visual Search)', 2.2, 0.9),
    (4.5, 0.4, C_GAME_M,  'Game\n(Attentional Spotter)', 2.2, 0.9),
    (7.5, 3.5, '#374151', 'RT (ms)\nAccuracy\nFalse Alarms', 2.0, 1.2),
    (7.5, 1.5, '#374151', 'RT (ms)\nAccuracy\nFalse Alarms', 2.0, 1.2),
]
for (x, y, col, txt, w, h) in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.08', linewidth=1.5,
        edgecolor='white', facecolor=col, alpha=0.88, zorder=3)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='white', zorder=4)

for (tx, ty) in [(4.5, 4.65), (4.5, 3.25)]:
    ax.annotate('', xy=(tx, ty), xytext=(3.0, 4.1),
        arrowprops=dict(arrowstyle='->', color=C_SINGLE, lw=1.8))

for (tx, ty) in [(4.5, 2.25), (4.5, 0.85)]:
    ax.annotate('', xy=(tx, ty), xytext=(3.0, 2.1),
        arrowprops=dict(arrowstyle='->', color=C_MULTI, lw=1.8))

for (sx, sy) in [(6.7,4.65),(6.7,3.25),(6.7,2.25),(6.7,0.85)]:
    ax.annotate('', xy=(7.5, 4.1 if sy>3 else 2.1),
        xytext=(sx, sy),
        arrowprops=dict(arrowstyle='->', color=GREY, lw=1.4))

ax.text(0.05, 5.5, 'Between-subjects\n(Target Load)', fontsize=9,
        color=GREY, style='italic')
ax.text(4.5, 5.5, 'Within-subjects\n(Modality)', fontsize=9,
        color=GREY, style='italic')
ax.text(7.5, 5.5, 'Dependent\nVariables', fontsize=9,
        color=GREY, style='italic')
ax.text(5.6, 5.0, 'Counterbalanced', fontsize=8, color=GREY,
        style='italic')

rqs = ['RQ1: Game ↔ Lab correlation?',
       'RQ2: Single vs Multiple difficulty?',
       'RQ3: Game vs Lab interface effect?',
       'RQ4: Practice/level effect?']
for i, rq in enumerate(rqs):
    ax.text(0.2 + i*2.45, 0.1, rq, fontsize=8, color='#374151',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#D1D5DB'))

save(fig, 'fig0_study_overview.png')

print("Fig 1 — RT distributions…")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#F9FAFB')
fig.suptitle('Reaction Time Distributions — All 4 Design Cells',
             **FONT_TITLE, y=1.01)

cells = [('single','lab',  C_LAB,    'Single Target — Lab Task'),
         ('single','game', C_GAME,   'Single Target — Game'),
         ('multiple','lab', C_LAB_M, 'Multiple Target — Lab Task'),
         ('multiple','game',C_GAME_M,'Multiple Target — Game')]

for ax, (load, mod, col, title) in zip(axes.flat, cells):
    rt = data[(data['target_load']==load) & (data['modality']==mod)]['rt_ms'].dropna()
    ax.hist(rt, bins=30, color=col, alpha=0.55, edgecolor='white',
            density=True, zorder=2, label='Histogram')
    if len(rt) > 5:
        kde = gaussian_kde(rt, bw_method=0.3)
        xr  = np.linspace(rt.min(), rt.max(), 300)
        ax.plot(xr, kde(xr), color=col, lw=2.5, zorder=3, label='KDE')

    ax.axvline(rt.mean(),   color='#111827', lw=2,   ls='--', zorder=4, label=f'Mean {rt.mean():.0f} ms')
    ax.axvline(rt.median(), color=GREY,      lw=1.5, ls=':',  zorder=4, label=f'Median {rt.median():.0f} ms')
    style_ax(ax, title=title, xlabel='RT (ms)', ylabel='Density')
    ax.legend(fontsize=8, framealpha=0.8)

    sw, psw = stats.shapiro(rt.sample(min(len(rt),50), random_state=1))
    txt = (f"n = {len(rt)}\nM = {rt.mean():.0f} ms\nSD = {rt.std():.0f} ms\n"
           f"Skew = {stats.skew(rt):.2f}\nSW p = {psw:.3f}")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#D1D5DB', alpha=0.9))

plt.tight_layout()
save(fig, 'fig1_rt_distributions.png')

print("Fig 2 — RQ1 Concurrent validity…")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#F9FAFB')
fig.suptitle('RQ1 — Concurrent Validity: Does the Game Measure the Same Thing as the Lab Task?',
             **FONT_TITLE, y=1.02)

for ax, load, col, n_exp in zip(axes, ['single','multiple'],
                                 [C_SINGLE, C_MULTI], [21, 16]):
    pivot = (ppt[ppt['target_load']==load]
             .pivot(index='participant', columns='modality', values='mean_rt')
             .dropna())
    if 'lab' not in pivot.columns or 'game' not in pivot.columns:
        continue

    ax.set_facecolor(LIGHT_GREY)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#D1D5DB')
    ax.grid(color='white', lw=1.2, zorder=0)

    ax.scatter(pivot['lab'], pivot['game'], color=col, s=80, alpha=0.85,
               edgecolors='white', lw=1.2, zorder=4)
    for pid, row in pivot.iterrows():
        ax.annotate(str(int(pid)), (row['lab'], row['game']),
                    fontsize=7, color='#374151',
                    xytext=(4, 4), textcoords='offset points')

    m, b = np.polyfit(pivot['lab'], pivot['game'], 1)
    xr   = np.linspace(pivot['lab'].min(), pivot['lab'].max(), 200)
    ax.plot(xr, m*xr+b, '--', color=col, lw=2, alpha=0.7, zorder=3)

    lims = [min(pivot['lab'].min(), pivot['game'].min()),
            max(pivot['lab'].max(), pivot['game'].max())]
    ax.plot(lims, lims, ':', color=GREY, lw=1.5, alpha=0.6, label='Perfect agreement (y=x)')

    r, p   = stats.pearsonr(pivot['lab'], pivot['game'])
    rho, _ = stats.spearmanr(pivot['lab'], pivot['game'])
    sig    = 'p < 0.05 ✓' if p < 0.05 else 'p = n.s. ✗'
    txt = f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\np = {p:.3f}  {sig}\nn = {len(pivot)}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va='top', bbox=dict(boxstyle='round,pad=0.5',
            facecolor='white', edgecolor='#D1D5DB', alpha=0.95))

    cond_label = load.capitalize()
    ax.set_title(f'{cond_label} Target Condition (n={len(pivot)})',
                 **FONT_TITLE, pad=8)
    ax.set_xlabel('Lab Task — Mean RT (ms)', **FONT_LABEL)
    ax.set_ylabel('Game — Mean RT (ms)', **FONT_LABEL)
    ax.legend(fontsize=9, framealpha=0.8)

plt.tight_layout()
save(fig, 'fig2_rq1_validity.png')

print("Fig 3 — RQ2 Single vs Multiple…")
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#F9FAFB')
fig.suptitle('RQ2 — Target Load Effect: Single vs Multiple Targets\n'
             '(Feature Integration Theory Prediction)',
             **FONT_TITLE, y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ax_a = fig.add_subplot(gs[0, :2])
conditions = [('single','lab'), ('single','game'),
              ('multiple','lab'), ('multiple','game')]
xlabels    = ['Single\nLab', 'Single\nGame', 'Multiple\nLab', 'Multiple\nGame']
colors     = [C_LAB, C_GAME, C_LAB_M, C_GAME_M]

for i, ((load, mod), col, lbl) in enumerate(zip(conditions, colors, xlabels)):
    g    = ppt[(ppt['target_load']==load) & (ppt['modality']==mod)]['mean_rt']
    m    = g.mean(); hw = ci95(g)

    ax_a.bar(i, m, color=col, alpha=0.75, edgecolor='white',
             width=0.6, zorder=2)
    ax_a.errorbar(i, m, yerr=hw, color='#111827', capsize=6,
                  capthick=2, lw=2, zorder=5)

    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(g))
    ax_a.scatter(np.full(len(g), i) + jitter, g.values,
                 color='white', s=30, alpha=0.9, edgecolors=col,
                 lw=1.2, zorder=6)

    ax_a.text(i, m + hw + 150, f'{m:.0f}', ha='center',
              fontsize=10, fontweight='bold', color=col)

def sig_bracket(ax, x1, x2, y, text, dy=200):
    ax.plot([x1, x1, x2, x2], [y, y+dy, y+dy, y], lw=1.5, color='#111827')
    ax.text((x1+x2)/2, y+dy+50, text, ha='center', fontsize=10,
            fontweight='bold', color='#111827')

sig_bracket(ax_a, 0, 2, 8200, '*** p < 0.001\nd = 4.94', 350)
sig_bracket(ax_a, 1, 3, 4200, '*** p < 0.001\nd = 2.14', 350)

style_ax(ax_a, title='Mean RT per Condition ± 95% CI\n(dots = individual participants)',
         ylabel='Mean RT (ms)')
ax_a.set_xticks(range(4)); ax_a.set_xticklabels(xlabels, fontsize=11)
ax_a.set_ylim(0, 10000)

ax_b = fig.add_subplot(gs[0, 2])
effects = {'Lab\n(Single vs Multi)': 4.94,
           'Game\n(Single vs Multi)': 2.14}
ecols   = [C_LAB, C_GAME]
bars = ax_b.barh(list(effects.keys()), list(effects.values()),
                 color=ecols, alpha=0.82, edgecolor='white', height=0.5)
for bar, val in zip(bars, effects.values()):
    ax_b.text(val + 0.05, bar.get_y() + bar.get_height()/2,
              f'd = {val:.2f}', va='center', fontsize=11,
              fontweight='bold', color='#111827')
ax_b.axvline(0.2, ls=':', color=GREY, lw=1.2, alpha=0.7, label='Small (0.2)')
ax_b.axvline(0.5, ls=':', color=GREY, lw=1.2, alpha=0.7, label='Medium (0.5)')
ax_b.axvline(0.8, ls=':', color=GREY, lw=1.2, alpha=0.7, label='Large (0.8)')
style_ax(ax_b, title="Effect Size (Cohen's d)\nSingle vs Multiple",
         xlabel="Cohen's d")
ax_b.legend(fontsize=8)
ax_b.set_xlim(0, 6)

ax_c = fig.add_subplot(gs[1, :2])
single_lab = ppt[(ppt['target_load']=='single') &
                 (ppt['modality']=='lab')].set_index('participant')['mean_rt']
multi_lab  = ppt[(ppt['target_load']=='multiple') &
                 (ppt['modality']=='lab')].set_index('participant')['mean_rt']

x_s = np.zeros(len(single_lab))
x_m = np.ones(len(multi_lab))
ax_c.scatter(x_s, single_lab.values, color=C_LAB, s=60, alpha=0.85,
             edgecolors='white', lw=1, zorder=4, label='Single (n=21)')
ax_c.scatter(x_m, multi_lab.values,  color=C_LAB_M, s=60, alpha=0.85,
             edgecolors='white', lw=1, zorder=4, label='Multiple (n=16)')
ax_c.scatter([0], [single_lab.mean()], marker='D', s=150, color=C_LAB,
             edgecolors='#111827', lw=2, zorder=6)
ax_c.scatter([1], [multi_lab.mean()],  marker='D', s=150, color=C_LAB_M,
             edgecolors='#111827', lw=2, zorder=6)
ax_c.plot([0, 1], [single_lab.mean(), multi_lab.mean()],
          '--', color='#111827', lw=2, alpha=0.5)
ax_c.text(0, single_lab.mean()-600, f'M = {single_lab.mean():.0f} ms',
          ha='center', fontsize=10, fontweight='bold', color=C_LAB)
ax_c.text(1, multi_lab.mean()+300, f'M = {multi_lab.mean():.0f} ms',
          ha='center', fontsize=10, fontweight='bold', color=C_LAB_M)
style_ax(ax_c, title='Lab Task RT — Individual Participants\n(♦ = group mean)',
         ylabel='Mean RT (ms)')
ax_c.set_xticks([0,1])
ax_c.set_xticklabels(['Single Target', 'Multiple Target'], fontsize=12)
ax_c.legend(fontsize=9)
ax_c.text(0.5, 0.97,
          f'Δ = +{multi_lab.mean()-single_lab.mean():.0f} ms  |  '
          f't = −13.02, p < 0.001, d = 4.94',
          transform=ax_c.transAxes, ha='center', va='top', fontsize=10,
          color='#111827',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEFCE8',
                    edgecolor='#FCD34D'))

ax_d = fig.add_subplot(gs[1, 2])
ax_d.set_facecolor(LIGHT_GREY)
ax_d.spines[['top','right']].set_visible(False)
ax_d.grid(color='white', lw=1.2)
n_targets = [1, 5]
rt_means  = [single_lab.mean(), multi_lab.mean()]
rt_cis    = [ci95(ppt[(ppt['target_load']=='single') &
                      (ppt['modality']=='lab')]['mean_rt']),
             ci95(ppt[(ppt['target_load']=='multiple') &
                      (ppt['modality']=='lab')]['mean_rt'])]
ax_d.plot(n_targets, rt_means, 'o-', color=C_LAB_M, lw=2.5,
          markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=4)
ax_d.fill_between(n_targets,
                  [m-e for m,e in zip(rt_means,rt_cis)],
                  [m+e for m,e in zip(rt_means,rt_cis)],
                  color=C_LAB_M, alpha=0.15)
ax_d.set_xticks([1, 5])
ax_d.set_xticklabels(['1 target\n(Single)', '5 targets\n(Multiple)'], fontsize=10)
ax_d.set_ylabel('Lab Mean RT (ms)', **FONT_LABEL)
ax_d.set_title('Feature Integration Theory:\nRT scales with target load',
               **FONT_TITLE, pad=8)
ax_d.text(0.5, 0.1,
          'Treisman & Gelade (1980)\nSerial attentional search',
          transform=ax_d.transAxes, ha='center', fontsize=9,
          style='italic', color=GREY)

save(fig, 'fig3_rq2_target_load.png')

print("Fig 4 — RQ3 Modality effect…")
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.patch.set_facecolor('#F9FAFB')
fig.suptitle('RQ3 — Modality Effect: Does the Game Alter Performance Compared to the Lab Task?',
             **FONT_TITLE, y=1.02)

for ax, load in zip(axes, ['single', 'multiple']):
    ax.set_facecolor(LIGHT_GREY)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#D1D5DB')
    ax.grid(axis='y', color='white', lw=1.2, zorder=0)

    piv = (ppt[ppt['target_load']==load]
           .pivot(index='participant', columns='modality', values='mean_rt')
           .dropna())
    if 'lab' not in piv.columns or 'game' not in piv.columns:
        continue

    col_l = C_LAB   if load=='single' else C_LAB_M
    col_g = C_GAME  if load=='single' else C_GAME_M

    for pid, row in piv.iterrows():
        alpha = 0.35
        color = '#93C5FD' if load=='single' else '#FCA5A5'
        ax.plot([0, 1], [row['lab'], row['game']], '-', color=color,
                lw=1.2, alpha=alpha, zorder=2)

    m_lab  = piv['lab'].mean();  hw_l = ci95(piv['lab'])
    m_game = piv['game'].mean(); hw_g = ci95(piv['game'])
    ax.plot([0, 1], [m_lab, m_game], 'o-', color='#111827',
            lw=3, markersize=12, markeredgecolor='white',
            markeredgewidth=2, zorder=5)
    ax.errorbar(0, m_lab,  yerr=hw_l, color=col_l, capsize=8, capthick=2.5,
                lw=2.5, zorder=6)
    ax.errorbar(1, m_game, yerr=hw_g, color=col_g, capsize=8, capthick=2.5,
                lw=2.5, zorder=6)
    ax.scatter([0, 1], [m_lab, m_game], s=140,
               color=[col_l, col_g], edgecolors='white',
               lw=2, zorder=7)

    ax.text(0, m_lab  + hw_l + 200, f'{m_lab:.0f} ms',
            ha='center', fontsize=11, fontweight='bold', color=col_l)
    ax.text(1, m_game + hw_g + 200, f'{m_game:.0f} ms',
            ha='center', fontsize=11, fontweight='bold', color=col_g)

    diff = piv['lab'] - piv['game']
    t_val, p_val = stats.ttest_rel(piv['lab'], piv['game'])
    d    = diff.mean() / diff.std()
    direction = 'Lab faster' if diff.mean() > 0 else 'Game faster'

    ax.text(0.5, 0.97,
            f'Δ = {diff.mean():+.0f} ms  ({direction})\n'
            f'Paired t({len(piv)-1}) = {t_val:.2f},  p < 0.001\n'
            f"Cohen's d = {d:.2f}  (large)",
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEFCE8',
                      edgecolor='#FCD34D', alpha=0.95))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Lab Task', 'Game'], fontsize=13)
    ax.set_ylabel('Mean RT (ms)', **FONT_LABEL)
    ax.set_title(f'{load.capitalize()} Target Condition\n'
                 f'(n = {len(piv)} matched participants)', **FONT_TITLE)

plt.tight_layout()
save(fig, 'fig4_rq3_modality.png')

print("Fig 5 — RQ4 Practice effect…")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.patch.set_facecolor('#F9FAFB')
fig.suptitle('RQ4 — Practice & Level Effects: How Does RT Change Across Trials?',
             **FONT_TITLE, y=1.01)

panels = [('lab',  'single',   C_LAB,    'Lab Task — Single Target\n(Expect: practice ↓ RT)'),
          ('lab',  'multiple', C_LAB_M,  'Lab Task — Multiple Target\n(Expect: practice ↓ RT)'),
          ('game', 'single',   C_GAME,   'Game — Single Target\n(Expect: difficulty ↑ RT)'),
          ('game', 'multiple', C_GAME_M, 'Game — Multiple Target\n(Expect: mixed)')]

for ax, (mod, load, col, title) in zip(axes.flat, panels):
    lv = (data[(data['modality']==mod) & (data['target_load']==load)]
          .groupby('level')['rt_ms']
          .agg(M='mean', SE='sem', N='count')
          .reset_index())

    ax.set_facecolor(LIGHT_GREY)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#D1D5DB')
    ax.grid(axis='y', color='white', lw=1.2, zorder=0)

    for pid, g in data[(data['modality']==mod) & (data['target_load']==load)].groupby('participant'):
        d = g.sort_values('level')
        ax.plot(d['level'], d['rt_ms'], '-', color=col,
                alpha=0.12, lw=1, zorder=1)

    ax.plot(lv['level'], lv['M'], 'o-', color=col, lw=2.5,
            markersize=6, markeredgecolor='white', markeredgewidth=1.2,
            zorder=4, label='Group mean')
    ax.fill_between(lv['level'], lv['M']-lv['SE'], lv['M']+lv['SE'],
                    color=col, alpha=0.2, zorder=3)

    valid = lv.dropna(subset=['M'])
    if len(valid) >= 3:
        slope, intercept, r, p, _ = stats.linregress(valid['level'], valid['M'])
        xr = np.linspace(valid['level'].min(), valid['level'].max(), 100)
        ls = '--' if p < 0.05 else ':'
        ax.plot(xr, slope*xr + intercept, ls, color='#111827',
                lw=1.8, alpha=0.8, zorder=5,
                label=f'Trend (r={r:.2f}, p={p:.3f})')

    style_ax(ax, title=title,
             xlabel='Level' if mod=='game' else 'Trial Number',
             ylabel='RT (ms)')
    ax.legend(fontsize=8, framealpha=0.85)

plt.tight_layout()
save(fig, 'fig5_rq4_practice.png')

print("Fig 6 — Accuracy and false alarms…")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('#F9FAFB')
fig.suptitle('Accuracy and Error Metrics Across Conditions',
             **FONT_TITLE, y=1.02)

ax = axes[0]
for i, (load, mod, col) in enumerate([('single','lab',C_LAB),
                                        ('single','game',C_GAME),
                                        ('multiple','lab',C_LAB_M),
                                        ('multiple','game',C_GAME_M)]):
    g  = ppt[(ppt['target_load']==load) & (ppt['modality']==mod)]['mean_acc']
    m  = g.mean(); hw = ci95(g)
    ax.bar(i, m*100, color=col, alpha=0.82, edgecolor='white',
           width=0.65, zorder=2)
    ax.errorbar(i, m*100, yerr=hw*100, color='#111827',
                capsize=5, capthick=2, lw=1.8, zorder=5)
    ax.text(i, m*100 - 1.5, f'{m*100:.1f}%', ha='center',
            fontsize=10, fontweight='bold', color='white', zorder=6)
style_ax(ax, title='Mean Accuracy (%) per Cell',
         ylabel='Accuracy (%)')
ax.set_ylim(85, 103)
ax.set_xticks(range(4))
ax.set_xticklabels(['SIN\nLab','SIN\nGame','MUL\nLab','MUL\nGame'], fontsize=9)
ax.axhline(100, color=GREY, lw=1, ls='--', alpha=0.5)

ax = axes[1]
fa_s = ppt[(ppt['target_load']=='single')   & (ppt['modality']=='game')]['sum_fa']
fa_m = ppt[(ppt['target_load']=='multiple') & (ppt['modality']=='game')]['sum_fa']
bw = 0.35
x  = np.arange(2)
bars_s = ax.bar(x[0], fa_s.mean(), bw, color=C_GAME,   alpha=0.82, edgecolor='white', label='Single')
bars_m = ax.bar(x[1], fa_m.mean(), bw, color=C_GAME_M, alpha=0.82, edgecolor='white', label='Multiple')
ax.errorbar(x[0], fa_s.mean(), yerr=ci95(fa_s), color='#111827',
            capsize=6, capthick=2, lw=2, zorder=5)
ax.errorbar(x[1], fa_m.mean(), yerr=ci95(fa_m), color='#111827',
            capsize=6, capthick=2, lw=2, zorder=5)

jitter_s = np.random.default_rng(0).uniform(-0.1, 0.1, len(fa_s))
jitter_m = np.random.default_rng(1).uniform(-0.1, 0.1, len(fa_m))
ax.scatter(np.zeros(len(fa_s))+jitter_s, fa_s.values,
           color='white', edgecolors=C_GAME, s=35, lw=1.2, alpha=0.85, zorder=6)
ax.scatter(np.ones(len(fa_m))+jitter_m,  fa_m.values,
           color='white', edgecolors=C_GAME_M, s=35, lw=1.2, alpha=0.85, zorder=6)
style_ax(ax, title='False Alarms — Game Modality\n(mean ± 95% CI)',
         ylabel='Total False Alarms per Participant')
ax.set_xticks([0,1])
ax.set_xticklabels(['Single\nTarget', 'Multiple\nTarget'], fontsize=11)
ax.legend(fontsize=9)

ax = axes[2]
gm    = data[(data['modality']=='game') & (data['target_load']=='multiple')]
fa_lv = gm.groupby('level')['false_alarms'].agg(['mean','sem'])
ax.bar(fa_lv.index, fa_lv['mean'], color=C_GAME_M, alpha=0.82,
       edgecolor='white', zorder=2)
ax.errorbar(fa_lv.index, fa_lv['mean'], yerr=fa_lv['sem'],
            color='#111827', fmt='none', capsize=4, capthick=1.5, lw=1.5, zorder=5)
style_ax(ax, title='False Alarms by Level\n(Game, Multiple Target)',
         xlabel='Level', ylabel='Mean False Alarms ± SE')

plt.tight_layout()
save(fig, 'fig6_accuracy_false_alarms.png')

print("Fig 7 — Normality summary…")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('#F9FAFB')
fig.suptitle('Data Quality & Normality — Justification for Non-Parametric / Log-Transform',
             **FONT_TITLE, y=1.02)

cells_qq = [('single','lab',C_LAB,'Single/Lab'),
            ('single','game',C_GAME,'Single/Game'),
            ('multiple','lab',C_LAB_M,'Multi/Lab'),
            ('multiple','game',C_GAME_M,'Multi/Game')]

ax = axes[0]
ax.set_facecolor(LIGHT_GREY)
ax.spines[['top','right']].set_visible(False)
ax.grid(color='white', lw=1.2)
for load, mod, col, lbl in cells_qq:
    rt = data[(data['target_load']==load) & (data['modality']==mod)]['rt_ms'].dropna()
    (osm, osr), _ = stats.probplot(rt, dist='norm')
    ax.plot(osm, osr, 'o', color=col, markersize=3, alpha=0.5, label=lbl)
ax.plot([-3,3], [ax.get_ylim()[0], ax.get_ylim()[1]], '--k', lw=1, alpha=0.4)
style_ax(ax, title='Q-Q Plots — All 4 Cells\n(points should follow diagonal)',
         xlabel='Theoretical Quantiles', ylabel='Sample Quantiles')
ax.legend(fontsize=8, markerscale=2)

ax = axes[1]
norm_data = []
for (pid, load, mod), g in data.groupby(['participant','target_load','modality']):
    rt = g['rt_ms'].dropna()
    if len(rt) < 3: continue
    _, p = stats.shapiro(rt)
    norm_data.append({'PID': pid, 'Condition': f'{load[:3]}/{mod}', 'SW_p': p})
nd = pd.DataFrame(norm_data)
pivot_sw = nd.pivot(index='PID', columns='Condition', values='SW_p').fillna(1)
im = ax.imshow(pivot_sw.values, aspect='auto', cmap='RdYlGn',
               vmin=0, vmax=0.3)
ax.set_xticks(range(len(pivot_sw.columns)))
ax.set_xticklabels(pivot_sw.columns, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(pivot_sw.index)))
ax.set_yticklabels(pivot_sw.index, fontsize=7)
ax.axhline(20.5, color='white', lw=2)
ax.set_title('Shapiro-Wilk p-values\n(Green = normal, Red = not normal)',
             **FONT_TITLE, pad=8)
plt.colorbar(im, ax=ax, label='p-value')
ax.text(-0.5, 10, 'Single', rotation=90, va='center', fontsize=9, color=C_SINGLE)
ax.text(-0.5, 28, 'Multiple', rotation=90, va='center', fontsize=9, color=C_MULTI)

ax = axes[2]
rt_all = data['rt_ms'].dropna()
rt_log = np.log(rt_all[rt_all > 0])
ax.set_facecolor(LIGHT_GREY)
ax.spines[['top','right']].set_visible(False)
ax.grid(color='white', lw=1.2)
ax2b = ax.twinx()
ax.hist(rt_all,  bins=40, color=C_LAB,   alpha=0.5, density=True,
        label='Raw RT', zorder=2)
ax2b.hist(rt_log, bins=40, color=C_GAME, alpha=0.5, density=True,
          label='Log(RT)', zorder=2)
ax.set_xlabel('RT (ms) / log(RT)', **FONT_LABEL)
ax.set_ylabel('Density (raw)', fontsize=11, color=C_LAB)
ax2b.set_ylabel('Density (log)', fontsize=11, color=C_GAME)
ax.set_title('Raw vs Log-Transformed RT\n(Log-transform reduces skew)',
             **FONT_TITLE, pad=8)
lines = [mpatches.Patch(color=C_LAB,  alpha=0.6, label='Raw RT'),
         mpatches.Patch(color=C_GAME, alpha=0.6, label='log(RT)')]
ax.legend(handles=lines, fontsize=9)
sw_raw, p_raw = stats.shapiro(rt_all.sample(50, random_state=1))
sw_log, p_log = stats.shapiro(rt_log.sample(50, random_state=1))
ax.text(0.98, 0.97,
        f'Raw: SW p = {p_raw:.4f}\nLog: SW p = {p_log:.4f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#D1D5DB'))

plt.tight_layout()
save(fig, 'fig7_normality_quality.png')

print("Fig 8 — Summary dashboard…")
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#111827')
fig.suptitle('Attention Task Validation — Key Findings Summary',
             fontsize=18, fontweight='bold', color='white', y=0.98)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.45)

dark_bg  = '#1F2937'
text_col = '#F9FAFB'

def dark_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(dark_bg)
    for sp in ax.spines.values():
        sp.set_color('#374151')
    ax.tick_params(colors=text_col, labelsize=9)
    ax.grid(axis='y', color='#374151', lw=1, zorder=0)
    if title:  ax.set_title(title,  color=text_col, fontsize=11, fontweight='bold', pad=7)
    if xlabel: ax.set_xlabel(xlabel, color='#9CA3AF', fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color='#9CA3AF', fontsize=9)

kpis = [
    ('4900 ms', 'Lab RT difference\n(Multi − Single)', C_LAB_M),
    ('d = 4.94', 'Effect size\n(largest in study)', C_LAB),
    ('r = 0.29', 'Game–Lab correlation\n(Single, n.s.)', C_GAME),
    ('26 / 74', 'Participants with\nsignificant trend', C_MULTI),
]
for i, (val, lbl, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(col)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis('off')
    ax.text(0.5, 0.62, val, ha='center', va='center',
            fontsize=24, fontweight='bold', color='white')
    ax.text(0.5, 0.25, lbl, ha='center', va='center',
            fontsize=10, color='white', alpha=0.9)

ax_rt = fig.add_subplot(gs[1, :2])
labels = ['Single\nLab', 'Single\nGame', 'Multi\nLab', 'Multi\nGame']
cols   = [C_LAB, C_GAME, C_LAB_M, C_GAME_M]
means  = [ppt[(ppt['target_load']==l) & (ppt['modality']==m)]['mean_rt'].mean()
          for l, m in [('single','lab'),('single','game'),
                       ('multiple','lab'),('multiple','game')]]
cis    = [ci95(ppt[(ppt['target_load']==l) & (ppt['modality']==m)]['mean_rt'])
          for l, m in [('single','lab'),('single','game'),
                       ('multiple','lab'),('multiple','game')]]
bars   = ax_rt.bar(range(4), means, color=cols, alpha=0.9,
                   edgecolor='#374151', width=0.6)
ax_rt.errorbar(range(4), means, yerr=cis, fmt='none',
               color=text_col, capsize=6, capthick=2, lw=2)
for i, (m, col) in enumerate(zip(means, cols)):
    ax_rt.text(i, m + 250, f'{m:.0f}', ha='center',
               fontsize=10, fontweight='bold', color=text_col)
dark_ax(ax_rt, title='Mean RT ± 95% CI — All 4 Conditions',
        ylabel='Mean RT (ms)')
ax_rt.set_xticks(range(4)); ax_rt.set_xticklabels(labels, color=text_col, fontsize=10)

ax_prac = fig.add_subplot(gs[1, 2:])
for load, col in [('single', C_LAB), ('multiple', C_LAB_M)]:
    lv = (data[(data['modality']=='lab') & (data['target_load']==load)]
          .groupby('level')['rt_ms'].agg(M='mean', SE='sem').reset_index())
    ax_prac.plot(lv['level'], lv['M'], 'o-', color=col, lw=2,
                 markersize=5, markeredgecolor='#111827',
                 markeredgewidth=1, label=load.capitalize(), zorder=4)
    ax_prac.fill_between(lv['level'], lv['M']-lv['SE'], lv['M']+lv['SE'],
                         color=col, alpha=0.18, zorder=3)
dark_ax(ax_prac, title='Lab Task: RT across Trials (Practice Effect)',
        xlabel='Trial Number', ylabel='Mean RT (ms)')
ax_prac.legend(fontsize=9, labelcolor=text_col,
               facecolor=dark_bg, edgecolor='#374151')

save(fig, 'fig8_summary_dashboard.png')

print("\nAll figures saved!")
print("Files: fig0_study_overview.png through fig8_summary_dashboard.png")