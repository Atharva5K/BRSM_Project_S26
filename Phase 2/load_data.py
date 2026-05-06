"""
BRSM Project Report 2 — Data Loading Pipeline
===============================================
Loads all 37 participants across 4 categories:
  - single/lab    (PIDs 1–21,  PsychoPy visual search, 1 target)
  - single/phone  (PIDs 1–21,  Attentional Spotter,    singleTarget)
  - multiple/lab  (PIDs 22–37, PsychoPy visual search, 5 targets)
  - multiple/phone(PIDs 22–37, Attentional Spotter,    multipleTargets)

Outputs
-------
df_trials        : trial/level-level data  (for regression, reliability)
df_participants  : per-participant means   (for ANOVA, t-tests, correlation)

Both are saved as CSV and also returned by load_all_data() for import.

RT definitions
--------------
Lab   — rt_first_ms : time to FIRST click  (single: same as last; multiple: first target found)
        rt_last_ms  : time to LAST click   (= total trial completion time)
Phone — rt_first_ms : InitialResponseTime(ms)   (time to first tap)
        avg_inter_target_ms : AvgInterTargetTime(ms) (multiple only, else NaN)

Incomplete game levels (Completed=False) are excluded.
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURE THIS PATH — everything else is derived from it
# ─────────────────────────────────────────────────────────────
BASE = os.path.join(
    os.path.expanduser('~'),
    'Sem4', 'BRSM', 'Project',
    'Attention Task Validation', 'data_brsm'
)


# ═════════════════════════════════════════════════════════════
# 1. LAB FILE LOADER
# ═════════════════════════════════════════════════════════════

def load_lab_file(filepath, participant_id, target_load):
    """
    Parse one PsychoPy CSV and return a tidy trial-level DataFrame.

    Parameters
    ----------
    filepath       : str   — absolute path to the CSV
    participant_id : int   — numeric PID extracted from filename
    target_load    : str   — 'single' or 'multiple'

    Returns
    -------
    DataFrame with columns:
        participant_id, target_load, modality,
        trial_num, target_col,
        rt_first_ms, rt_last_ms, accuracy, n_targets_clicked
    """
    df = pd.read_csv(filepath)
    trials = df[df['target_col'].notna()].copy()

    if len(trials) == 0:
        print(f"  WARNING: no trial rows found in {filepath}")
        return pd.DataFrame()

    # ── RT column detection (two PsychoPy export variants) ──
    if 'trials.mouse.time' in trials.columns:
        rt_col      = 'trials.mouse.time'
        clicked_col = 'trials.mouse.clicked_name'
    else:
        rt_col      = 'mouse.time'
        clicked_col = 'mouse.clicked_name'

    # ── Parse mouse.time list → first and last click times ──
    def parse_rt(val):
        """Return (first_rt, last_rt) in seconds, or (NaN, NaN)."""
        if pd.isna(val):
            return np.nan, np.nan
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list) and len(parsed) > 0:
                floats = [float(x) for x in parsed]
                return floats[0], floats[-1]
            v = float(parsed)
            return v, v
        except Exception:
            return np.nan, np.nan

    rt_parsed          = trials[rt_col].apply(parse_rt)
    trials['rt_first'] = rt_parsed.apply(lambda x: x[0])
    trials['rt_last']  = rt_parsed.apply(lambda x: x[1])

    # ── Count target clicks per trial ──
    def count_targets(clicked):
        if pd.isna(clicked):
            return 0
        try:
            parsed = ast.literal_eval(str(clicked))
            if isinstance(parsed, list):
                return sum(1 for x in parsed if 'target' in str(x).lower())
            return 1 if 'target' in str(clicked).lower() else 0
        except Exception:
            return 1 if 'target' in str(clicked).lower() else 0

    trials['n_targets_clicked'] = trials[clicked_col].apply(count_targets)

    # ── Accuracy: proportion of required targets clicked ──
    n_required = 5 if target_load == 'multiple' else 1
    trials['accuracy'] = (trials['n_targets_clicked'] / n_required).clip(0, 1)

    return pd.DataFrame({
        'participant_id'    : participant_id,
        'target_load'       : target_load,
        'modality'          : 'lab',
        'trial_num'         : trials['trials.thisN'].values,
        'target_col'        : trials['target_col'].values,
        'rt_first_ms'       : trials['rt_first'].values * 1000,
        'rt_last_ms'        : trials['rt_last'].values  * 1000,
        'accuracy'          : trials['accuracy'].values,
        'n_targets_clicked' : trials['n_targets_clicked'].values,
        'false_alarms'      : np.nan,                     # not measured in lab
        'avg_inter_target_ms': np.nan,                    # not applicable
    })


# ═════════════════════════════════════════════════════════════
# 2. PHONE FILE LOADER
# ═════════════════════════════════════════════════════════════

def load_phone_file(filepath, participant_id, target_load):
    """
    Parse one Attentional Spotter CSV and return a tidy level-level DataFrame.
    Incomplete levels (Completed=False) are excluded.

    Returns
    -------
    DataFrame with columns:
        participant_id, target_load, modality,
        trial_num (= Level), target_col (NaN — not logged),
        rt_first_ms, rt_last_ms (= rt_first_ms, no last-click in phone),
        accuracy, n_targets_clicked (NaN),
        false_alarms, avg_inter_target_ms
    """
    df = pd.read_csv(filepath)

    # ── Exclude incomplete levels ──
    n_total      = len(df)
    df           = df[df['Completed'] == True].copy()
    n_incomplete = n_total - len(df)
    if n_incomplete > 0:
        print(f"  INFO: PID {participant_id} ({target_load}/phone) — "
              f"excluded {n_incomplete} incomplete level(s)")

    if len(df) == 0:
        print(f"  WARNING: no complete levels in {filepath}")
        return pd.DataFrame()

    # ── Truncate to first 15 levels (handles PID 6 who replayed) ──
    if len(df) > 15:
        print(f"  INFO: PID {participant_id} ({target_load}/phone) — "
              f"truncating {len(df)} completed levels to first 15")
        df = df.head(15)

    # Phone only logs InitialResponseTime — treat as both first and last
    # for single-target (semantically equivalent).
    # For multiple-target, rt_last_ms is left as NaN (no equivalent exists;
    # use avg_inter_target_ms for that operationalisation instead).
    is_multiple = (target_load == 'multiple')

    return pd.DataFrame({
        'participant_id'     : participant_id,
        'target_load'        : target_load,
        'modality'           : 'phone',
        'trial_num'          : df['Level'].values,
        'target_col'         : np.nan,                       # not logged
        'rt_first_ms'        : df['InitialResponseTime(ms)'].values.astype(float),
        'rt_last_ms'         : np.nan if is_multiple
                               else df['InitialResponseTime(ms)'].values.astype(float),
        'accuracy'           : df['HitRate(%)'].values / 100.0,
        'n_targets_clicked'  : np.nan,
        'false_alarms'       : df['FalseAlarms'].values.astype(float),
        'avg_inter_target_ms': df['AvgInterTargetTime(ms)'].values.astype(float)
                               if is_multiple
                               else np.nan,
    })


# ═════════════════════════════════════════════════════════════
# 3. BATCH LOADER — iterate over all files
# ═════════════════════════════════════════════════════════════

def _extract_pid(filename):
    """Extract leading integer from filename, e.g. '22_visual...' → 22."""
    return int(filename.split('_')[0])


def load_all_data(base=BASE, verbose=True):
    """
    Load every participant file and return (df_trials, df_participants).
    """
    trial_dfs = []

    categories = [
        ('single',   'lab',   'lab'),
        ('single',   'phone', 'phone'),
        ('multiple', 'lab',   'lab'),
        ('multiple', 'phone', 'phone'),
    ]

    for target_load, folder, modality in categories:
        dirpath = os.path.join(base, target_load, folder)
        if not os.path.isdir(dirpath):
            print(f"  ERROR: directory not found: {dirpath}")
            continue

        files = sorted(
            f for f in os.listdir(dirpath)
            if f.endswith('.csv')
        )

        if verbose:
            print(f"\n{'─'*55}")
            print(f"Loading {target_load}/{folder}  ({len(files)} files)")
            print(f"{'─'*55}")

        for fname in files:
            fpath = os.path.join(dirpath, fname)
            pid   = _extract_pid(fname)

            try:
                if modality == 'lab':
                    df = load_lab_file(fpath, pid, target_load)
                else:
                    df = load_phone_file(fpath, pid, target_load)

                if df is not None and len(df) > 0:
                    trial_dfs.append(df)
                    if verbose:
                        print(f"  PID {pid:>3} | {len(df):>3} trials/levels loaded")
            except Exception as e:
                print(f"  ERROR loading {fname}: {e}")

    # ── Concatenate all trial-level data ──
    df_trials = pd.concat(trial_dfs, ignore_index=True)
    df_trials['target_load'] = pd.Categorical(
        df_trials['target_load'], categories=['single', 'multiple'], ordered=True
    )

    if verbose:
        print(f"\n{'═'*55}")
        print(f"df_trials shape : {df_trials.shape}")
        print(f"Participants    : {df_trials['participant_id'].nunique()}")
        print(f"{'═'*55}")

    # ── Build per-participant summary ──
    df_participants = _build_participant_summary(df_trials)

    if verbose:
        print(f"\ndf_participants shape : {df_participants.shape}")
        print(df_participants.to_string(index=False))

    return df_trials, df_participants


# ═════════════════════════════════════════════════════════════
# 4. PER-PARTICIPANT SUMMARY
# ═════════════════════════════════════════════════════════════

def _build_participant_summary(df_trials):
    """
    Collapse trial-level data to one row per (participant, modality).
    This is the unit of analysis for all inferential tests.
    """
    rows = []

    for (pid, load, mod), grp in df_trials.groupby(
            ['participant_id', 'target_load', 'modality']):

        rt_first  = grp['rt_first_ms'].dropna()
        rt_last   = grp['rt_last_ms'].dropna()
        acc       = grp['accuracy'].dropna()
        fa        = grp['false_alarms'].dropna()
        ait       = grp['avg_inter_target_ms'].dropna()

        row = {
            'participant_id'       : pid,
            'target_load'          : load,
            'modality'             : mod,
            'n_trials'             : len(rt_first),

            # RT — first click
            'mean_rt_first_ms'     : rt_first.mean()   if len(rt_first) > 0 else np.nan,
            'sd_rt_first_ms'       : rt_first.std()    if len(rt_first) > 1 else np.nan,
            'log_mean_rt_first'    : np.log(rt_first.mean()) if len(rt_first) > 0 else np.nan,

            # RT — last click (all targets found; NaN for phone/multiple)
            'mean_rt_last_ms'      : rt_last.mean()    if len(rt_last)  > 0 else np.nan,
            'sd_rt_last_ms'        : rt_last.std()     if len(rt_last)  > 1 else np.nan,
            'log_mean_rt_last'     : np.log(rt_last.mean()) if len(rt_last) > 0 else np.nan,

            # Accuracy
            'mean_accuracy'        : acc.mean()        if len(acc)      > 0 else np.nan,

            # Phone-only metrics
            'total_false_alarms'   : fa.sum()          if len(fa)       > 0 else np.nan,
            'mean_avg_inter_target_ms': ait.mean()     if len(ait)      > 0 else np.nan,
            'log_mean_avg_inter_target': np.log(ait.mean()) if len(ait) > 0 else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df['target_load'] = pd.Categorical(
        df['target_load'], categories=['single', 'multiple'], ordered=True
    )
    df = df.sort_values(['target_load', 'participant_id', 'modality']).reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════
# 5. MAIN — run standalone to validate and save CSVs
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df_trials, df_participants = load_all_data(verbose=True)

    # ── Save outputs ──
    out_dir = os.path.dirname(os.path.abspath(__file__))
    trials_path = os.path.join(out_dir, 'df_trials.csv')
    parts_path  = os.path.join(out_dir, 'df_participants.csv')

    df_trials.to_csv(trials_path, index=False)
    df_participants.to_csv(parts_path, index=False)

    print(f"\nSaved: {trials_path}")
    print(f"Saved: {parts_path}")

    # ── Quick sanity checks ──
    print("\n── Sanity checks ──")

    # Each participant should appear in exactly 2 rows (lab + phone)
    counts = df_participants.groupby('participant_id')['modality'].count()
    bad    = counts[counts != 2]
    if len(bad) == 0:
        print("✓ Every participant has exactly 2 modality rows (lab + phone)")
    else:
        print(f"✗ Participants with unexpected row count:\n{bad}")

    # Single group: PIDs 1–21; Multiple group: PIDs 22–37
    single_pids   = set(df_participants[df_participants['target_load']=='single']['participant_id'])
    multiple_pids = set(df_participants[df_participants['target_load']=='multiple']['participant_id'])
    print(f"✓ Single group  PIDs: {sorted(single_pids)}")
    print(f"✓ Multiple group PIDs: {sorted(multiple_pids)}")

    # Trial counts
    print("\n── Trial/level counts per cell ──")
    summary = df_trials.groupby(['target_load', 'modality'])['participant_id'].agg(
        n_participants='nunique',
        total_trials='count'
    )
    print(summary.to_string())

    print("\nDone.")
