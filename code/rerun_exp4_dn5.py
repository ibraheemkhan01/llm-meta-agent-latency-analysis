"""
Re-run ONLY the dn=5 Orchestrator+LLM case from Experiment 4 with per-cycle
timing saved. Replaces the corrupted dn=5 row in exp4_frequency_ablation.csv
and writes exp4_dn5_cycle_timing.csv for outlier inspection.

Usage:
    python rerun_exp4_dn5.py
"""

import os, sys, time, json, csv, asyncio, re
import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_overhead_backtest import (
    load_intraday_bars, clip_to_window,
    InstrumentedOrchestrator, run_orchestrator_timed,
    OUT_DIR,
    WINDOW_START, WINDOW_END,
)
from daily_backtest import SLIPPAGE_BPS, COMMISSION, STARTING, RESULTS_DIR

TRADING_DAYS_PER_YEAR = 252
DN = 5   # ← the only cadence we're re-running

OUTLIER_THRESHOLD_MS = 30_000   # calls longer than 30 s are network outliers


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    bars_all = load_intraday_bars()
    if not bars_all:
        print("No data — check connection."); return
    bars = clip_to_window(bars_all, WINDOW_START, WINDOW_END)
    if not bars:
        print("No bars in window."); return

    sample = list(bars.values())[0]
    print(f"\nWindow: {sample.index[0].date()} → {sample.index[-1].date()}  "
          f"({len(sample)} bars/ticker, {len(bars)} tickers)")

    # ── Run dn=5 Orchestrator ────────────────────────────────────────────────
    cadence_min = DN * 15
    label = f"Orchestrator+LLM (every {cadence_min} min)"
    print(f"\nRunning {label} with per-cycle timing...\n")

    orch = InstrumentedOrchestrator()
    r = run_orchestrator_timed(label, orch, bars, decision_n=DN)
    r["decision_n"] = DN

    # ── Save raw per-cycle timing ────────────────────────────────────────────
    cycle_df = pd.DataFrame(orch.timing_log)
    cycle_path = os.path.join(OUT_DIR, "exp4_dn5_cycle_timing.csv")
    cycle_df.to_csv(cycle_path, index=False)
    print(f"\nPer-cycle timing saved → {cycle_path}")

    # ── Outlier analysis ─────────────────────────────────────────────────────
    llm_ms = cycle_df["t_llm_call"] * 1000
    n_total   = len(llm_ms)
    n_outlier = (llm_ms > OUTLIER_THRESHOLD_MS).sum()
    median_ms = llm_ms.median()
    mean_raw  = llm_ms.mean()
    llm_clean = llm_ms[llm_ms <= OUTLIER_THRESHOLD_MS]
    mean_clean = llm_clean.mean() if len(llm_clean) > 0 else mean_raw

    print(f"\nOutlier analysis (threshold: {OUTLIER_THRESHOLD_MS} ms)")
    print(f"  Total calls   : {n_total}")
    print(f"  Outliers (>{OUTLIER_THRESHOLD_MS/1000:.0f}s): {n_outlier}")
    print(f"  Median        : {median_ms:.0f} ms")
    print(f"  Mean (raw)    : {mean_raw:.0f} ms")
    print(f"  Mean (cleaned): {mean_clean:.0f} ms")

    if n_outlier > 0:
        print(f"\n  Outlier timestamps:")
        for _, row in cycle_df[llm_ms > OUTLIER_THRESHOLD_MS].iterrows():
            print(f"    ts={row['timestamp']}  llm={row['t_llm_call']*1000:.0f}ms")

    # ── Rebuild summary row with clean stats ─────────────────────────────────
    # Recompute all timing fields excluding outlier calls
    clean_mask = (cycle_df["t_llm_call"] * 1000) <= OUTLIER_THRESHOLD_MS
    clean_log  = cycle_df[clean_mask]

    # Update the result row with clean timing
    if len(clean_log) > 0:
        r["t_llm_mean"]   = round(clean_log["t_llm_call"].mean() * 1000, 2)
        r["t_llm_p95"]    = round(clean_log["t_llm_call"].quantile(0.95) * 1000, 2)
        r["t_total_mean"] = round(clean_log["t_total"].mean() * 1000, 2)
        r["t_total_p95"]  = round(clean_log["t_total"].quantile(0.95) * 1000, 2)
        r["t_agent_mean"] = round(clean_log["t_agent_calls"].mean() * 1000, 2)
        r["t_prompt_mean"]= round(clean_log["t_prompt_build"].mean() * 1000, 2)
        r["t_parse_mean"] = round(clean_log["t_parse"].mean() * 1000, 2)
        r["llm_calls_clean"] = int(len(clean_log))
        r["llm_calls_outlier"] = int(n_outlier)

    # ── Update exp4_frequency_ablation.csv ──────────────────────────────────
    exp4_path = os.path.join(OUT_DIR, "exp4_frequency_ablation.csv")
    if os.path.exists(exp4_path):
        exp4_df = pd.read_csv(exp4_path)
        # Drop old dn=5 Orchestrator row
        mask_remove = (exp4_df["decision_n"] == DN) & (exp4_df["agent"].str.contains("Orchestrator"))
        exp4_df = exp4_df[~mask_remove]
        # Append new clean row
        new_row = pd.DataFrame([r])
        exp4_df = pd.concat([exp4_df, new_row], ignore_index=True)
        # Sort by decision_n then agent name for readability
        exp4_df = exp4_df.sort_values(["decision_n", "agent"]).reset_index(drop=True)
        exp4_df.to_csv(exp4_path, index=False)
        print(f"\nUpdated → {exp4_path}  (replaced dn=5 Orchestrator row with clean data)")
    else:
        print(f"\nexp4_frequency_ablation.csv not found — saving standalone result.")
        pd.DataFrame([r]).to_csv(os.path.join(OUT_DIR, "exp4_dn5_result.csv"), index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  AnnReturn : {r['ann_return']:+.4f}%")
    print(f"  t_llm_mean (clean): {r['t_llm_mean']:.0f} ms")
    print(f"  t_llm_p95  (clean): {r['t_llm_p95']:.0f} ms")
    print(f"  Calls (clean/total): {r.get('llm_calls_clean', r['llm_calls'])}/{r['llm_calls']}")
    print(f"  Total LLM cost: ${r['llm_cost_usd']:.5f}")


if __name__ == "__main__":
    main()
