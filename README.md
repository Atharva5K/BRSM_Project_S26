# Attention Task Validation — BRSM Project

> **Can a mobile game replace a lab-based visual search task for measuring selective attention?**

This repository contains the analysis code and report for a **Behavioural Research Statistics & Methods (BRSM)** project that validates a gamified attention assessment tool (*Attentional Spotter*) against a traditional PsychoPy-based lab visual search paradigm.

---

## Study Overview

| Aspect | Details |
|--------|---------|
| **Design** | 2 × 2 Mixed Factorial |
| **Between-subjects** | Target Load — *Single* (n = 21) vs *Multiple* (n = 16) |
| **Within-subjects** | Modality — *Lab Task* (PsychoPy) vs *Game* (Attentional Spotter) |
| **Participants** | 37 total (PIDs 1–21: Single, PIDs 22–37: Multiple) |
| **Dependent Variables** | Reaction Time (ms), Accuracy (%), False Alarms |

```
                ┌─── Lab Task (Visual Search) ──────┐
  Single Target ┤                                    ├─▶ RT, Accuracy, FA
   (n = 21)     └─── Game (Attentional Spotter) ────┘
                ┌─── Lab Task (Visual Search) ──────┐
Multiple Target ┤                                    ├─▶ RT, Accuracy, FA
   (n = 16)     └─── Game (Attentional Spotter) ────┘
```

---

## Research Questions

| # | Research Question | Statistical Method |
|---|-------------------|--------------------|
| **RQ1** | Concurrent Validity — Does the game measure the same construct as the lab task? | Pearson *r*, Spearman *ρ* |
| **RQ2** | Target Load Effect — Does multiple-target search take longer (Feature Integration Theory)? | Welch's *t*-test, Cohen's *d* |
| **RQ3** | Modality Effect — Does the game interface alter attentional performance? | Paired *t*-test |
| **RQ4** | Practice / Level Effect — Does RT change across successive trials? | Pearson *r*, Spearman *ρ* (level vs RT) |

---

## Installation

### Prerequisites

- **Python 3.8+**
- Required packages:

```bash
pip install pandas numpy matplotlib scipy
```

### Data Setup

Place the raw data files inside `Attention Task Validation/data_brsm/` following the directory structure shown above. Each CSV should be named with the participant ID as a prefix (e.g., `1_visual_search.csv`, `22_attentional_spotter.csv`).

---

## Usage

Run both scripts from the **project root directory** (`BRSM/Code/`):

### 1. Descriptive Statistics & Hypothesis Tests

```bash
python descriptive_stats.py
```

**What it does:**
- Loads and cleans all lab (PsychoPy) and game (Attentional Spotter) CSV data
- Computes per-participant and per-cell summary statistics (M, SD, Median, IQR, CI)
- Performs **Shapiro-Wilk** normality tests for each participant × condition
- Runs **RQ1**: Pearson & Spearman correlations for concurrent validity
- Runs **RQ2**: Welch's *t*-tests for single vs multiple target load (with Cohen's *d*)
- Runs **RQ3**: Paired *t*-tests for lab vs game modality effect
- Runs **RQ4**: Level-by-RT correlation analysis for practice effects
- Generates a combined multi-panel diagnostic figure
- Exports a report-ready summary CSV

**Outputs:**
| File | Description |
|------|-------------|
| `descriptive_stats_plots.png` | Multi-panel figure (boxplots, bar charts, scatter, Q-Q plots) |
| `descriptive_stats_summary.csv` | Per-participant mean RT, accuracy, and false alarms |

---

### 2. Research Visualizations

```bash
python visualizations.py
```

**What it does:**
- Produces **8 publication-quality PNG figures**, one per research question or analysis theme
- Includes scatter plots with regression lines, paired-sample dot plots, bar charts with individual data points, Q-Q plots, KDE overlays, and a dark-themed summary dashboard

**Outputs:**

| Figure | Description |
|--------|-------------|
| `fig0_study_overview.png` | Study design schematic (2 × 2 mixed-factorial layout) |
| `fig1_rt_distributions.png` | RT histograms + KDE for all 4 design cells |
| `fig2_rq1_validity.png` | RQ1 — Lab vs Game RT scatter (concurrent validity) |
| `fig3_rq2_target_load.png` | RQ2 — Target load effect with Cohen's *d* & FIT illustration |
| `fig4_rq3_modality.png` | RQ3 — Paired lab vs game RT with individual trajectories |
| `fig5_rq4_practice.png` | RQ4 — RT across levels/trials (practice effects) |
| `fig6_accuracy_false_alarms.png` | Accuracy & false alarm comparison across conditions |
| `fig7_normality_quality.png` | Q-Q plots, Shapiro-Wilk heatmap, log-transform comparison |
| `fig8_summary_dashboard.png` | Dark-themed KPI dashboard for presentations |

---

## Methodology

### Data Loading Pipeline

Both scripts share a common data loading strategy:

1. **Lab data** — PsychoPy `visual_search` CSVs are parsed, extracting RT from the last mouse click time (list-string format `'[1.82, 6.79]'`) and computing accuracy as the proportion of correctly clicked targets (1 for single, 5 for multiple).
2. **Game data** — Attentional Spotter CSVs are loaded directly with columns like `InitialResponseTime(ms)`, `SuccessRate(%)`, `FalseAlarms`, and `HitRate(%)`.
3. All data is unified into a single DataFrame with consistent columns: `participant`, `target_load`, `modality`, `level`, `rt_ms`, `accuracy`, `false_alarms`, etc.

### Statistical Analyses

| Analysis | Method | Implementation |
|----------|--------|----------------|
| Descriptive statistics | M, SD, Median, IQR, 95% CI (*t*-distribution) | `pandas` aggregation |
| Normality testing | Shapiro-Wilk test | `scipy.stats.shapiro` |
| Correlation (RQ1) | Pearson *r*, Spearman *ρ* | `scipy.stats.pearsonr`, `spearmanr` |
| Group comparison (RQ2) | Welch's independent *t*-test | `scipy.stats.ttest_ind` |
| Paired comparison (RQ3) | Paired-samples *t*-test | `scipy.stats.ttest_rel` |
| Trend analysis (RQ4) | Level–RT correlation | `scipy.stats.pearsonr`, `linregress` |
| Effect size | Cohen's *d* (pooled SD) | Custom `cohens_d()` function |

---

## Report

The full project report is included as **`BRSM_Project_1 (1).pdf`**, containing:
- Literature review and theoretical framework (Feature Integration Theory)
- Detailed results for all four research questions
- Discussion of concurrent validity findings
- Limitations and future directions

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3** | Core language |
| **pandas** | Data wrangling and aggregation |
| **NumPy** | Numerical operations |
| **Matplotlib** | All visualizations and figures |
| **SciPy** | Statistical tests (Shapiro-Wilk, *t*-tests, correlations) |
| **PsychoPy** | Lab task stimulus presentation *(data collection only)* |

---

## License

This project was developed for academic coursework (BRSM — Semester 4). All rights reserved.

---

<p align="center">
  <i>Built for cognitive science research</i>
</p>
