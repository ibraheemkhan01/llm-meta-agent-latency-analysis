"""
Compute overhead experiments for second paper:
  "Where does time and compute go in LLM-orchestrated multi-agent decision systems,
   and when is the orchestration overhead justified by decision quality?"

Experiments:
  1. Per-stage latency breakdown across all architectures
  2. Agent count scaling: Orchestrator with 1 / 2 / 3 strategy agents
  3. Universe scaling: 10 / 50 / 100 / 149 tickers
  4. Decision frequency ablation: decision_n = 1 / 3 / 5 / 15

All experiments use:
  - 45-day intraday 15-min bars (Mar 6 – Apr 17 2026)
  - Slippage: 10 bps per leg + $1 commission (via daily_backtest constants)
  - gpt-4o-mini with token-cost tracking ($0.15/1M input, $0.60/1M output)

Results saved to backtest_results/compute_overhead/
"""

import os, sys, time, json, csv, asyncio
import numpy as np
import pandas as pd
import yfinance as yf

if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daily_backtest import (
    UCBBanditAgent, LLMUCBAgent, EnsembleAgent,
    build_observation, TICKERS, STARTING, RESULTS_DIR,
    SLIPPAGE_BPS, COMMISSION,
)
from strategy_agents.trend_agent import TrendAgent
from strategy_agents.momentum_agent import MomentumAgent
from strategy_agents.mean_reversion_agent import MeanReversionAgent
from llm.orchestrator import Orchestrator

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_START  = "2026-03-06"
WINDOW_END    = "2026-04-18"
OUT_DIR       = os.path.join(RESULTS_DIR, "compute_overhead")

GPT_INPUT_COST_PER_M  = 0.150   # $ per 1M input tokens
GPT_OUTPUT_COST_PER_M = 0.600   # $ per 1M output tokens

TRADING_DAYS_PER_YEAR = 252
BARS_PER_DAY_15MIN    = 26      # ~6.5 h × 4 bars/h

# ── Data loading ─────────────────────────────────────────────────────────────

def load_intraday_bars(tickers=None) -> dict:
    tickers = tickers or TICKERS
    print(f"Downloading 60d 15-min bars for {len(tickers)} tickers...")
    raw = yf.download(
        tickers=tickers, period="60d", interval="15m",
        group_by="ticker", progress=False, auto_adjust=True,
    )
    bars = {}
    for ticker in tickers:
        try:
            df = raw[ticker][["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df = df.dropna()
            if not df.empty:
                bars[ticker] = df
        except Exception:
            continue
    return bars


def clip_to_window(bars: dict, start: str, end: str) -> dict:
    result = {}
    for ticker, df in bars.items():
        idx = df.index
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if idx.tzinfo is not None:
            s = s.tz_localize(idx.tzinfo) if s.tzinfo is None else s.tz_convert(idx.tzinfo)
            e = e.tz_localize(idx.tzinfo) if e.tzinfo is None else e.tz_convert(idx.tzinfo)
        sub = df[(idx >= s) & (idx < e)]
        if not sub.empty:
            result[ticker] = sub
    return result


# ── Instrumented Orchestrator ─────────────────────────────────────────────────

class InstrumentedOrchestrator(Orchestrator):
    """
    Wraps Orchestrator.analyze() with per-stage wall-clock timing and
    token-cost tracking. Each call appends one record to self.timing_log.
    """

    def __init__(self, strategy_agents=None):
        super().__init__(strategy_agents=strategy_agents)
        self.timing_log: list = []   # one dict per decision cycle
        self._reasoning_path = os.path.join(OUT_DIR, "llm_reasoning.jsonl")

    def analyze(self, observation: pd.DataFrame, timestamp=None) -> list:
        import re

        record = {
            "timestamp":        str(timestamp),
            "n_tickers":        len(observation),
            "n_agents":         len(self.strategy_agents),
            "t_feature":        0.0,   # feature compute already done before analyze()
            "t_agent_calls":    0.0,
            "t_prompt_build":   0.0,
            "t_llm_call":       0.0,
            "t_parse":          0.0,
            "t_total":          0.0,
            "prompt_tokens":    0,
            "completion_tokens":0,
            "llm_cost_usd":     0.0,
            "n_trades":         0,
        }

        t0 = time.perf_counter()

        # Stage 1: strategy agent calls
        t1 = time.perf_counter()
        agent_recs = asyncio.run(self._call_strategy_agents(observation))
        record["t_agent_calls"] = time.perf_counter() - t1

        # Stage 2: prompt construction
        t2 = time.perf_counter()
        prompt = self._generate_prompt(observation, agent_recs, timestamp=timestamp)
        record["t_prompt_build"] = time.perf_counter() - t2

        # Stage 3: LLM call (sync, with usage capture)
        t3 = time.perf_counter()
        llm_response, usage = self._call_llm_with_usage(prompt)
        record["t_llm_call"] = time.perf_counter() - t3
        if usage:
            record["prompt_tokens"]     = usage.prompt_tokens
            record["completion_tokens"] = usage.completion_tokens
            record["llm_cost_usd"] = (
                usage.prompt_tokens     * GPT_INPUT_COST_PER_M  / 1_000_000 +
                usage.completion_tokens * GPT_OUTPUT_COST_PER_M / 1_000_000
            )

        # Stage 4: parse
        t4 = time.perf_counter()
        parsed = None
        if isinstance(llm_response, str):
            m = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    pass
        record["t_parse"] = time.perf_counter() - t4

        record["t_total"] = time.perf_counter() - t0

        trades = parsed.get("trades", []) if parsed else []
        record["n_trades"] = len(trades)

        # agent attribution + allocation log (from parent)
        bias    = parsed.get("overall_bias", "unknown") if parsed else "unknown"
        summary = parsed.get("summary", "") if parsed else ""
        agent_counts = {"TrendAgent": 0, "MomentumAgent": 0, "MeanReversionAgent": 0}
        for tr in trades:
            ag = self._infer_agent(tr.get("reason", ""))
            tr["_agent_source"] = ag
            agent_counts[ag] = agent_counts.get(ag, 0) + 1
        total = sum(agent_counts.values()) or 1
        from datetime import datetime as _dt
        import csv as _csv
        ts_str   = str(timestamp) if timestamp is not None else _dt.now().isoformat()
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "backtest_results", "orchestrator_allocation_log.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_hdr = not os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if write_hdr:
                w.writerow(["timestamp","total_trades","trend_n","momentum_n","mr_n",
                            "trend_pct","momentum_pct","mr_pct","bias","summary"])
            w.writerow([ts_str, total,
                        agent_counts["TrendAgent"], agent_counts["MomentumAgent"],
                        agent_counts["MeanReversionAgent"],
                        round(agent_counts["TrendAgent"]/total*100,1),
                        round(agent_counts["MomentumAgent"]/total*100,1),
                        round(agent_counts["MeanReversionAgent"]/total*100,1),
                        bias, summary[:120]])

        self.timing_log.append(record)
        return trades

    def _call_llm_with_usage(self, prompt: str):
        """Synchronous LLM call that also returns the usage object."""
        import asyncio

        async def _call():
            try:
                resp = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content":
                         "You are a precise, risk-aware portfolio strategist who coordinates "
                         "multiple specialized trading agents and selects the best strategy "
                         "for current market conditions."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.65,
                    max_tokens=1000,
                )
                return resp.choices[0].message.content.strip(), resp.usage
            except Exception as e:
                return f"LLM Error: {e}", None

        return asyncio.run(_call())


# ── Instrumented run_agent ────────────────────────────────────────────────────

def run_agent_timed(name: str, agent, bars: dict, decision_n: int = 15) -> dict:
    """
    Like daily_backtest.run_agent but records per-cycle timing for
    non-LLM agents (feature compute + agent call).
    Returns the standard result dict plus timing fields.
    """
    sim = {"balance": STARTING, "positions": {}, "transactions": []}
    all_ts = sorted(set(ts for df in bars.values() for ts in df.index))

    feature_times = []
    agent_times   = []

    for i, ts in enumerate(all_ts):
        prices = {t: float(bars[t].loc[ts,"close"]) for t in bars if ts in bars[t].index}

        for ticker in list(sim["positions"]):
            pos = sim["positions"][ticker]
            p   = prices.get(ticker)
            if p is None:
                continue
            if p >= pos["exit_target"]:
                p_fill = p * (1 - SLIPPAGE_BPS / 10_000)
                pnl = (p_fill - pos["entry_price"]) * pos["quantity"] - COMMISSION
                sim["balance"] += p_fill * pos["quantity"] - COMMISSION
                sim["transactions"].append({"type": "TAKE PROFIT", "pnl": pnl})
                if hasattr(agent, "update"):
                    agent.update("TAKE PROFIT", ticker)
                del sim["positions"][ticker]
            elif p <= pos["stop_loss"]:
                p_fill = p * (1 - SLIPPAGE_BPS / 10_000)
                pnl = (p_fill - pos["entry_price"]) * pos["quantity"] - COMMISSION
                sim["balance"] += p_fill * pos["quantity"] - COMMISSION
                sim["transactions"].append({"type": "STOP LOSS", "pnl": pnl})
                if hasattr(agent, "update"):
                    agent.update("STOP LOSS", ticker)
                del sim["positions"][ticker]

        if i % decision_n == 0:
            tf0 = time.perf_counter()
            obs = build_observation(bars, ts)
            feature_times.append(time.perf_counter() - tf0)

            if obs.empty:
                continue

            ta0 = time.perf_counter()
            try:
                ideas = agent.analyze(obs)
            except Exception:
                ideas = []
            agent_times.append(time.perf_counter() - ta0)

            candidates = ideas if isinstance(ideas, list) else []
            for idea in candidates[:5]:
                ticker = idea.get("ticker")
                if not ticker or ticker in sim["positions"]:
                    continue
                ep = float(idea.get("entry_price", prices.get(ticker, 0)))
                if ep <= 0:
                    continue
                ep_fill = ep * (1 + SLIPPAGE_BPS / 10_000)
                qty = int((sim["balance"] * 0.01) / ep_fill)
                if qty <= 0 or qty * ep_fill + COMMISSION > sim["balance"]:
                    continue
                sim["balance"] -= qty * ep_fill + COMMISSION
                sim["positions"][ticker] = {
                    "quantity":    qty,
                    "entry_price": ep_fill,
                    "exit_target": ep_fill * 1.035,
                    "stop_loss":   ep_fill * 0.965,
                }

    # liquidation (no slippage — theoretical)
    for ticker, pos in sim["positions"].items():
        p   = prices.get(ticker, pos["entry_price"])
        pnl = (p - pos["entry_price"]) * pos["quantity"]
        sim["balance"] += p * pos["quantity"]
        sim["transactions"].append({"type": "LIQUIDATION", "pnl": pnl})

    nav    = sim["balance"]
    ret    = (nav - STARTING) / STARTING * 100
    closed = [t for t in sim["transactions"] if t["type"] in ("TAKE PROFIT", "STOP LOSS")]
    exp    = float(np.mean([t["pnl"] for t in closed])) if closed else 0.0

    n_days   = len(set(ts.date() if hasattr(ts, "date") else ts for ts in all_ts))
    ann_ret  = ret * (TRADING_DAYS_PER_YEAR / max(n_days, 1))

    return {
        "agent":        name,
        "nav":          round(nav, 2),
        "return_pct":   round(ret, 4),
        "ann_return":   round(ann_ret, 4),
        "trades":       len(sim["transactions"]),
        "expectancy":   round(exp, 2),
        "t_feature_mean": round(np.mean(feature_times) * 1000, 2) if feature_times else 0,
        "t_agent_mean":   round(np.mean(agent_times)   * 1000, 2) if agent_times   else 0,
        "t_feature_p95":  round(np.percentile(feature_times, 95) * 1000, 2) if feature_times else 0,
        "t_agent_p95":    round(np.percentile(agent_times,   95) * 1000, 2) if agent_times   else 0,
        "t_llm_mean":     0.0,
        "t_llm_p95":      0.0,
        "prompt_tokens":  0,
        "llm_cost_usd":   0.0,
    }


def run_orchestrator_timed(name: str, agent: InstrumentedOrchestrator,
                           bars: dict, decision_n: int = 15) -> dict:
    """Run the instrumented orchestrator and merge timing from its log."""
    sim = {"balance": STARTING, "positions": {}, "transactions": []}
    all_ts = sorted(set(ts for df in bars.values() for ts in df.index))
    feature_times = []

    for i, ts in enumerate(all_ts):
        prices = {t: float(bars[t].loc[ts,"close"]) for t in bars if ts in bars[t].index}

        for ticker in list(sim["positions"]):
            pos = sim["positions"][ticker]
            p   = prices.get(ticker)
            if p is None:
                continue
            if p >= pos["exit_target"]:
                p_fill = p * (1 - SLIPPAGE_BPS / 10_000)
                pnl = (p_fill - pos["entry_price"]) * pos["quantity"] - COMMISSION
                sim["balance"] += p_fill * pos["quantity"] - COMMISSION
                sim["transactions"].append({"type": "TAKE PROFIT", "pnl": pnl})
                del sim["positions"][ticker]
            elif p <= pos["stop_loss"]:
                p_fill = p * (1 - SLIPPAGE_BPS / 10_000)
                pnl = (p_fill - pos["entry_price"]) * pos["quantity"] - COMMISSION
                sim["balance"] += p_fill * pos["quantity"] - COMMISSION
                sim["transactions"].append({"type": "STOP LOSS", "pnl": pnl})
                del sim["positions"][ticker]

        if i % decision_n == 0:
            tf0 = time.perf_counter()
            obs = build_observation(bars, ts)
            feature_times.append(time.perf_counter() - tf0)
            if obs.empty:
                continue
            try:
                ideas = agent.analyze(obs, timestamp=ts)
            except Exception:
                ideas = []
            candidates = ideas if isinstance(ideas, list) else []
            for idea in candidates[:5]:
                ticker = idea.get("ticker")
                if not ticker or ticker in sim["positions"]:
                    continue
                ep = float(idea.get("entry_price", prices.get(ticker, 0)))
                if ep <= 0:
                    continue
                ep_fill = ep * (1 + SLIPPAGE_BPS / 10_000)
                qty = int((sim["balance"] * 0.01) / ep_fill)
                if qty <= 0 or qty * ep_fill + COMMISSION > sim["balance"]:
                    continue
                sim["balance"] -= qty * ep_fill + COMMISSION
                sim["positions"][ticker] = {
                    "quantity":    qty,
                    "entry_price": ep_fill,
                    "exit_target": ep_fill * 1.035,
                    "stop_loss":   ep_fill * 0.965,
                }

    for ticker, pos in sim["positions"].items():
        p   = prices.get(ticker, pos["entry_price"])
        pnl = (p - pos["entry_price"]) * pos["quantity"]
        sim["balance"] += p * pos["quantity"]
        sim["transactions"].append({"type": "LIQUIDATION", "pnl": pnl})

    nav    = sim["balance"]
    ret    = (nav - STARTING) / STARTING * 100
    closed = [t for t in sim["transactions"] if t["type"] in ("TAKE PROFIT", "STOP LOSS")]
    exp    = float(np.mean([t["pnl"] for t in closed])) if closed else 0.0

    n_days  = len(set(ts.date() if hasattr(ts, "date") else ts for ts in all_ts))
    ann_ret = ret * (TRADING_DAYS_PER_YEAR / max(n_days, 1))

    log = agent.timing_log
    llm_times  = [r["t_llm_call"] for r in log]
    total_cost = sum(r["llm_cost_usd"] for r in log)
    avg_in_tok = np.mean([r["prompt_tokens"]     for r in log]) if log else 0
    avg_out_tok= np.mean([r["completion_tokens"] for r in log]) if log else 0

    return {
        "agent":          name,
        "nav":            round(nav, 2),
        "return_pct":     round(ret, 4),
        "ann_return":     round(ann_ret, 4),
        "trades":         len(sim["transactions"]),
        "expectancy":     round(exp, 2),
        "t_feature_mean": round(np.mean(feature_times) * 1000, 2) if feature_times else 0,
        "t_agent_mean":   round(np.mean([r["t_agent_calls"] for r in log]) * 1000, 2) if log else 0,
        "t_prompt_mean":  round(np.mean([r["t_prompt_build"] for r in log]) * 1000, 2) if log else 0,
        "t_llm_mean":     round(np.mean(llm_times) * 1000, 2) if llm_times else 0,
        "t_parse_mean":   round(np.mean([r["t_parse"] for r in log]) * 1000, 2) if log else 0,
        "t_total_mean":   round(np.mean([r["t_total"] for r in log]) * 1000, 2) if log else 0,
        "t_feature_p95":  round(np.percentile(feature_times, 95) * 1000, 2) if feature_times else 0,
        "t_agent_p95":    round(np.percentile([r["t_agent_calls"] for r in log], 95) * 1000, 2) if log else 0,
        "t_llm_p95":      round(np.percentile(llm_times, 95) * 1000, 2) if llm_times else 0,
        "t_total_p95":    round(np.percentile([r["t_total"] for r in log], 95) * 1000, 2) if log else 0,
        "prompt_tokens":  round(avg_in_tok, 1),
        "completion_tokens": round(avg_out_tok, 1),
        "llm_cost_usd":   round(total_cost, 6),
        "llm_calls":      len(log),
        "n_agents":       len(agent.strategy_agents),
    }


# ── Experiment runners ────────────────────────────────────────────────────────

def exp1_latency_breakdown(bars: dict):
    """All architectures, default cadence=15. Per-stage timing."""
    print("\n" + "="*65)
    print("EXPERIMENT 1: Per-stage latency breakdown")
    print("="*65)
    results = []

    non_llm = [
        ("TrendAgent",         TrendAgent()),
        ("MomentumAgent",      MomentumAgent()),
        ("MeanReversionAgent", MeanReversionAgent()),
        ("EnsembleAgent",      EnsembleAgent()),
        ("UCBBanditAgent",     UCBBanditAgent()),
    ]
    for name, agent in non_llm:
        print(f"  Running {name}...")
        r = run_agent_timed(name, agent, bars, decision_n=15)
        results.append(r)

    print("  Running Orchestrator+LLM (3 agents)...")
    orch = InstrumentedOrchestrator()
    r = run_orchestrator_timed("Orchestrator+LLM", orch, bars, decision_n=15)
    results.append(r)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "exp1_latency_breakdown.csv")
    pd.DataFrame(results).to_csv(out, index=False)

    # also save raw per-cycle timing
    raw_out = os.path.join(OUT_DIR, "exp1_cycle_timing.csv")
    pd.DataFrame(orch.timing_log).to_csv(raw_out, index=False)

    print(f"\n  Saved → {out}")
    _print_latency_table(results)
    return results


def exp2_agent_scaling(bars: dict):
    """Orchestrator with 1 / 2 / 3 strategy agents."""
    print("\n" + "="*65)
    print("EXPERIMENT 2: Agent count scaling")
    print("="*65)
    agent_pool = [TrendAgent(), MomentumAgent(), MeanReversionAgent()]
    results = []
    for n in [1, 2, 3]:
        subset = agent_pool[:n]
        label  = f"Orchestrator ({n} agent{'s' if n>1 else ''})"
        print(f"  Running {label}...")
        orch = InstrumentedOrchestrator(strategy_agents=subset)
        r    = run_orchestrator_timed(label, orch, bars, decision_n=15)
        results.append(r)

    out = os.path.join(OUT_DIR, "exp2_agent_scaling.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    _print_scaling_table(results, key_col="n_agents")
    return results


def exp3_universe_scaling(all_bars_full: dict):
    """Orchestrator on subsets of the ticker universe: 10/50/100/149."""
    print("\n" + "="*65)
    print("EXPERIMENT 3: Universe size scaling")
    print("="*65)
    tickers_sorted = sorted(all_bars_full.keys())
    results = []
    for n in [10, 50, 100, len(tickers_sorted)]:
        subset_tickers = tickers_sorted[:n]
        subset_bars    = {t: all_bars_full[t] for t in subset_tickers if t in all_bars_full}
        label = f"Orchestrator ({len(subset_bars)} tickers)"
        print(f"  Running {label}...")
        orch = InstrumentedOrchestrator()
        r    = run_orchestrator_timed(label, orch, subset_bars, decision_n=15)
        r["n_tickers"] = len(subset_bars)
        results.append(r)

    out = os.path.join(OUT_DIR, "exp3_universe_scaling.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    _print_scaling_table(results, key_col="n_tickers")
    return results


def exp4_frequency_ablation(bars: dict):
    """Orchestrator + all non-LLM agents at decision_n = 1 / 3 / 5 / 15."""
    print("\n" + "="*65)
    print("EXPERIMENT 4: Decision frequency ablation")
    print("="*65)
    results = []
    for dn in [1, 3, 5, 15]:
        cadence_min = dn * 15
        label_sfx   = f"(every {cadence_min} min)"

        # Non-LLM baselines at this cadence
        for name, AgentCls in [
            ("TrendAgent",         TrendAgent),
            ("EnsembleAgent",      EnsembleAgent),
        ]:
            lbl = f"{name} {label_sfx}"
            print(f"  Running {lbl}...")
            r = run_agent_timed(lbl, AgentCls(), bars, decision_n=dn)
            r["decision_n"] = dn
            results.append(r)

        # LLM orchestrator at this cadence
        lbl = f"Orchestrator+LLM {label_sfx}"
        print(f"  Running {lbl}...")
        orch = InstrumentedOrchestrator()
        r    = run_orchestrator_timed(lbl, orch, bars, decision_n=dn)
        r["decision_n"] = dn
        results.append(r)

    out = os.path.join(OUT_DIR, "exp4_frequency_ablation.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    _print_freq_table(results)
    return results


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def _print_latency_table(results):
    print(f"\n  {'Agent':<30} {'AnnRet%':>8} {'Exp$':>7} "
          f"{'t_feat':>7} {'t_agent':>8} {'t_llm':>8} {'t_total':>8} {'LLMcost$':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
    for r in results:
        print(f"  {r['agent']:<30} {r['ann_return']:>+7.2f}%  "
              f"${r['expectancy']:>+6.2f}  "
              f"{r.get('t_feature_mean',0):>6.0f}ms  "
              f"{r.get('t_agent_mean',0):>7.0f}ms  "
              f"{r.get('t_llm_mean',0):>7.0f}ms  "
              f"{r.get('t_total_mean',0):>7.0f}ms  "
              f"${r.get('llm_cost_usd',0):>8.5f}")


def _print_scaling_table(results, key_col):
    print(f"\n  {key_col:<12} {'t_agent':>8} {'t_llm':>8} {'t_total':>8} "
          f"{'tokens_in':>10} {'LLMcost$':>9} {'AnnRet%':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*8}")
    for r in results:
        print(f"  {r.get(key_col,'?'):<12} "
              f"{r.get('t_agent_mean',0):>7.0f}ms  "
              f"{r.get('t_llm_mean',0):>7.0f}ms  "
              f"{r.get('t_total_mean',0):>7.0f}ms  "
              f"{r.get('prompt_tokens',0):>9.0f}  "
              f"${r.get('llm_cost_usd',0):>8.5f}  "
              f"{r.get('ann_return',0):>+7.2f}%")


def _print_freq_table(results):
    print(f"\n  {'Agent':<35} {'dn':>4} {'calls':>6} {'AnnRet%':>8} "
          f"{'t_llm':>8} {'LLMcost$':>9}")
    print(f"  {'-'*35} {'-'*4} {'-'*6} {'-'*8} {'-'*8} {'-'*9}")
    for r in results:
        print(f"  {r['agent']:<35} {r.get('decision_n','-'):>4}  "
              f"{r.get('llm_calls',0):>5}  "
              f"{r['ann_return']:>+7.2f}%  "
              f"{r.get('t_llm_mean',0):>7.0f}ms  "
              f"${r.get('llm_cost_usd',0):>8.5f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data once, clip to window
    bars_all  = load_intraday_bars()
    if not bars_all:
        print("No data — check connection."); return
    bars = clip_to_window(bars_all, WINDOW_START, WINDOW_END)
    if not bars:
        print("No bars in window — yfinance 15-min data only goes back 60 days."); return

    sample = list(bars.values())[0]
    print(f"\nWindow: {sample.index[0].date()} → {sample.index[-1].date()}  "
          f"({len(sample)} bars/ticker, {len(bars)} tickers)")

    # Run all experiments
    exp1_latency_breakdown(bars)
    exp2_agent_scaling(bars)
    exp3_universe_scaling(bars)
    exp4_frequency_ablation(bars)

    print(f"\n{'='*65}")
    print(f"All results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
