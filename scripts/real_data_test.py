"""Test the simulator on real S&P 500 data.

Run from the repo root:   python scripts/real_data_test.py
Inputs : data/sp500_shiller_monthly.csv, data/sp500tr_yahoo_monthly.csv
Outputs: printed tables, results/real_data_results.md, figures/real_*.png

Steps
  1. Verify the data (Yahoo calendar-year returns vs published S&P 500 total returns).
  2. Calibrate GBM (mu, sigma) on two datasets; correct the Shiller averaging bias.
  3. Test the GBM assumptions (normality, autocorrelation, tails).
  4. Re-run the baseline experiment three ways on the same loan/budget:
       (a) GBM with calibrated parameters
       (b) every actual 20-year window of S&P 500 history (rolling backtest)
       (c) 12-month block bootstrap of history (fat tails, no GBM)
  5. Breakeven loan APR: where P(invest-first beats payoff-first) crosses 50%,
     GBM vs history, against the volatility-drag prediction r* ~ mu - sigma^2/2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradsim import INVEST_FIRST, PAYOFF_FIRST, BALANCED, Loan, Scenario, run_all  # noqa: E402
from gradsim.calibrate import estimate_gbm_params  # noqa: E402
from gradsim.historical import (backtest, block_bootstrap, gbm_diagnostics,  # noqa: E402
                                load_shiller_total_return, load_yahoo_total_return, rolling_windows)
from gradsim.metrics import crra_certainty_equivalent, prob_a_beats_b, summary  # noqa: E402

FIG, OUT = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

LOAN = Loan(principal=30_000, annual_rate=0.065, term_months=120)  # same placeholders as run_baseline.py
BUDGET, HORIZON, N_PATHS = 800, 240, 20_000
STRATS = [INVEST_FIRST, BALANCED, PAYOFF_FIRST]
PUBLISHED_TR = {2008: -37.00, 2022: -18.11, 2023: 26.29, 2024: 25.02}  # S&P 500 total return, % (S&P DJI)

lines: list[str] = []


def say(s: str = ""):
    print(s)
    lines.append(s)


def annual_based_params(gross) -> dict[str, float]:
    """mu, sigma from non-overlapping 12-month log returns (immune to monthly averaging bias)."""
    r = np.log(np.asarray(gross, dtype=float))
    k = r.size // 12
    ann = r[: k * 12].reshape(k, 12).sum(axis=1)
    g, sig = ann.mean(), ann.std(ddof=1)
    return {"mu": g + sig**2 / 2, "sigma": sig, "g": g, "mu_se": sig / np.sqrt(k), "years": k}


def main():
    shiller, yahoo = load_shiller_total_return(), load_yahoo_total_return()
    sh26 = shiller["1926":]

    # 1. data verification -------------------------------------------------------------
    say("## 1. Data verification (Yahoo ^SP500TR calendar-year returns vs published)")
    say("| Year | Computed | Published |\n|---|---|---|")
    for yr, pub in PUBLISHED_TR.items():
        say(f"| {yr} | {100 * (yahoo[str(yr)].prod() - 1):.2f}% | {pub:.2f}% |")

    # 2. calibration ---------------------------------------------------------------------
    say("\n## 2. GBM calibration (annualised)")
    say("| Dataset | Method | μ | SE(μ) | σ | g = μ − σ²/2 |\n|---|---|---|---|---|---|")
    cal = {}
    for label, g in [("Shiller 1926–2023 (monthly avg)", sh26), ("Shiller 1988–2023 (monthly avg)", shiller["1988-02":]),
                     ("Yahoo 1988–2023 (month-end)", yahoo[:"2023-06"]), ("Yahoo 1988–2024 (month-end)", yahoo)]:
        e = estimate_gbm_params(np.r_[1.0, np.cumprod(g.values)])
        say(f"| {label} | monthly MLE | {e['mu']:.3f} | {e['mu_se']:.3f} | {e['sigma']:.3f} | {e['mu'] - e['sigma']**2 / 2:.3f} |")
    a = annual_based_params(sh26)
    say(f"| Shiller 1926–2023 | annual log returns | {a['mu']:.3f} | {a['mu_se']:.3f} | {a['sigma']:.3f} | {a['g']:.3f} |")
    cal["Long-run (1926–2023)"] = (a["mu"], a["sigma"])
    e = estimate_gbm_params(np.r_[1.0, np.cumprod(yahoo.values)])
    cal["Modern (1988–2024)"] = (e["mu"], e["sigma"])
    say("\nShiller prices are monthly *averages*: monthly σ is biased low by ≈√(2/3) (Working 1960). "
        "The annual-return estimator removes this; it is used for the long-run calibration.")

    # 3. diagnostics ---------------------------------------------------------------------
    say("\n## 3. Are monthly log returns i.i.d. normal (the GBM assumption)?")
    say("| Dataset | skew | excess kurt | Jarque–Bera | lag-1 autocorr (±2SE) | P(< −3σ) obs / normal | worst month |")
    say("|---|---|---|---|---|---|---|")
    for label, g in [("Shiller 1926–2023", sh26), ("Yahoo 1988–2024", yahoo)]:
        d = gbm_diagnostics(g)
        say(f"| {label} | {d['skew']:.2f} | {d['excess_kurtosis']:.2f} | {d['jarque_bera']:.0f} (5% crit 6.0) | "
            f"{d['lag1_autocorr']:.3f} (±{2 * d['ac1_se']:.3f}) | {d['freq_below_-3sd']:.2%} / 0.13% | {d['worst_month']:.1%} |")

    # 4. strategy comparison -------------------------------------------------------------
    scn0 = Scenario(LOAN, BUDGET, HORIZON)
    say(f"\n## 4. Baseline strategies on real data: ${LOAN.principal:,} @ {LOAN.annual_rate:.1%}, "
        f"${BUDGET}/mo, {HORIZON // 12}-year horizon")
    say("| Return model | Strategy | median | 5th pct | CE (γ=3) | P(invest-first > payoff-first) |")
    say("|---|---|---|---|---|---|")
    results = {}
    for label, (mu, sig) in cal.items():
        scn = Scenario(LOAN, BUDGET, HORIZON, mu=mu, sigma=sig)
        res = run_all(scn, STRATS, N_PATHS, seed=42)
        results[f"GBM {label}"] = {k: v.terminal_net_worth for k, v in res.items()}
    win = rolling_windows(sh26, HORIZON)
    results[f"History: {win.shape[0]} rolling 20y windows 1926–2023"] = {s.name: backtest(scn0, s, win) for s in STRATS}
    win71 = rolling_windows(shiller, HORIZON)
    results[f"History: {win71.shape[0]} rolling 20y windows 1871–2023"] = {s.name: backtest(scn0, s, win71) for s in STRATS}
    boot = block_bootstrap(sh26, HORIZON, N_PATHS, block=12, seed=42)
    results["Block bootstrap 1926–2023"] = {s.name: backtest(scn0, s, boot) for s in STRATS}
    for label, w in results.items():
        p = prob_a_beats_b(w[INVEST_FIRST.name], w[PAYOFF_FIRST.name])
        for i, s in enumerate(STRATS):
            st = summary(w[s.name])
            say(f"| {label if i == 0 else ''} | {s.name} | {st['median']:,.0f} | {st['p5']:,.0f} | "
                f"{crra_certainty_equivalent(w[s.name], 3):,.0f} | {f'{p:.1%}' if i == 0 else ''} |")

    # worst historical windows for invest-first relative to payoff-first
    diff = results[next(k for k in results if k.startswith("History"))]
    d = diff[INVEST_FIRST.name] - diff[PAYOFF_FIRST.name]
    starts = sh26.index[: win.shape[0]]
    worst = np.argsort(d)[:3]
    say("\nWorst historical start months for invest-first (vs payoff-first): " +
        ", ".join(f"{starts[i]:%Y-%m} ({d[i]:+,.0f})" for i in worst))
    say(f"Invest-first lost in {np.mean(d < 0):.1%} of windows; these windows overlap heavily "
        f"(≈{len(sh26) // HORIZON} independent 20-year periods), so treat this as descriptive.")

    # 5. breakeven APR ------------------------------------------------------------------
    rates = np.round(np.arange(0.04, 0.165, 0.01), 3)
    curves = {}
    for label, maker in [("GBM long-run", "gbm"), ("History rolling", "hist"), ("Bootstrap", "boot")]:
        ps = []
        for r in rates:
            scn = Scenario(Loan(LOAN.principal, r, 120), BUDGET, HORIZON, mu=cal["Long-run (1926–2023)"][0],
                           sigma=cal["Long-run (1926–2023)"][1])
            if maker == "gbm":
                res = run_all(scn, [INVEST_FIRST, PAYOFF_FIRST], 8000, seed=0)
                wi, wp = (res[k].terminal_net_worth for k in (INVEST_FIRST.name, PAYOFF_FIRST.name))
            else:
                paths = win if maker == "hist" else boot[:8000]
                wi, wp = backtest(scn, INVEST_FIRST, paths), backtest(scn, PAYOFF_FIRST, paths)
            ps.append(prob_a_beats_b(wi, wp))
        curves[label] = np.array(ps)
    mu, sig = cal["Long-run (1926–2023)"]
    say("\n## 5. Breakeven loan APR (P(invest-first wins) = 50%)")
    say(f"Volatility-drag prediction r* ≈ μ − σ²/2 = {mu - sig**2 / 2:.3f} (long-run calibration; nominal, continuous)")
    for label, p in curves.items():
        cross = np.interp(0.5, p[::-1], rates[::-1]) if p.min() < 0.5 < p.max() else float("nan")
        say(f"- {label}: breakeven APR ≈ {cross:.3f}")

    # figures -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, p in curves.items():
        ax.plot(rates * 100, p, "o-", label=label, ms=4)
    ax.axhline(0.5, color="k", lw=0.8)
    ax.axvline((mu - sig**2 / 2) * 100, color="gray", ls="--", lw=1, label="μ − σ²/2")
    ax.set(xlabel="Loan APR (%)", ylabel="P(invest-first beats payoff-first)",
           title="Breakeven loan rate: GBM vs actual S&P 500 history (20y horizon)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "real_breakeven_curve.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(starts, d / 1e3, lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="Start month of 20-year window", ylabel="Invest-first − payoff-first ($k)",
           title=f"Every historical 20-year window, 1926–2023 (\\${LOAN.principal // 1000}k @ {LOAN.annual_rate:.1%}, \\${BUDGET}/mo)")
    fig.tight_layout()
    fig.savefig(FIG / "real_rolling_windows.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    r = np.log(yahoo.values)
    z = (r - r.mean()) / r.std(ddof=1)
    ax.hist(z, bins=50, density=True, alpha=0.6, label="Yahoo ^SP500TR 1988–2024")
    x = np.linspace(-5, 4, 300)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "k", label="Normal (GBM)")
    ax.set(xlabel="Standardised monthly log return", ylabel="Density", yscale="log", ylim=(1e-3, 1),
           title="Fat left tail: real returns vs the GBM assumption")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "real_return_tails.png", dpi=150)

    (OUT / "real_data_results.md").write_text("# Real S&P 500 data test (generated by scripts/real_data_test.py)\n\n"
                                              + "\n".join(lines) + "\n")
    print(f"\nWrote {OUT / 'real_data_results.md'} and figures/real_*.png")


if __name__ == "__main__":
    main()
