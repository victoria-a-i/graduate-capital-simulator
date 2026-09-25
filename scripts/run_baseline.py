"""Baseline experiment: produces the first figures for the progress slide.

Run from the repo root:   python scripts/run_baseline.py
Outputs go to figures/.
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

from gradsim import DEFAULT_STRATEGIES, INVEST_FIRST, PAYOFF_FIRST, Loan, Scenario, Strategy, run_all  # noqa: E402
from gradsim.metrics import crra_certainty_equivalent, cvar, mc_standard_error, prob_a_beats_b, summary  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- Baseline assumptions (placeholders — replace with calibrated / sourced values) ----
LOAN = Loan(principal=30_000, annual_rate=0.065, term_months=120)
BUDGET = 800          # $/month for loan + investing
HORIZON = 240         # 20 years
MU, SIGMA = 0.08, 0.16
N_PATHS = 20_000


def base_scenario(**kw) -> Scenario:
    args = dict(loan=LOAN, monthly_budget=BUDGET, horizon_months=HORIZON, mu=MU, sigma=SIGMA)
    args.update(kw)
    return Scenario(**args)


def table():
    res = run_all(base_scenario(), DEFAULT_STRATEGIES, N_PATHS, seed=42)
    print(f"Loan ${LOAN.principal:,.0f} @ {LOAN.annual_rate:.1%}, min pmt ${LOAN.min_payment:,.0f}/mo, "
          f"budget ${BUDGET}/mo, horizon {HORIZON // 12}y, mu={MU}, sigma={SIGMA}\n")
    print(f"{'strategy':32s} {'mean':>10s} {'±SE':>7s} {'median':>10s} {'p5':>10s} {'CVaR5%':>10s} {'CE(g=3)':>10s} payoff")
    for name, r in res.items():
        w = r.terminal_net_worth
        s = summary(w)
        print(f"{name:32s} {s['mean']:10,.0f} {mc_standard_error(w):7,.0f} {s['median']:10,.0f} "
              f"{s['p5']:10,.0f} {cvar(w):10,.0f} {crra_certainty_equivalent(w, 3):10,.0f} "
              f"m{r.payoff_month}")
    p = prob_a_beats_b(res[INVEST_FIRST.name].terminal_net_worth, res[PAYOFF_FIRST.name].terminal_net_worth)
    print(f"\nP(invest-first ends richer than payoff-first) = {p:.1%}")
    return res


def fig_histogram(res):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    allw = np.concatenate([r.terminal_net_worth for r in res.values()])
    bins = np.linspace(np.percentile(allw, 0.5), np.percentile(allw, 99.5), 80)
    for name, r in res.items():
        ax.hist(r.terminal_net_worth / 1e3, bins=bins / 1e3, alpha=0.45, label=name)
    ax.set(xlabel="Net worth after 20 years ($k)", ylabel="Paths",
           title="Distribution of terminal net worth by strategy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "terminal_wealth_hist.png", dpi=150)


def fig_alpha_sweep():
    alphas = np.linspace(0, 1, 11)
    strats = [Strategy(f"a={a:.1f}", a) for a in alphas]
    res = run_all(base_scenario(), strats, N_PATHS, seed=42)
    med = [np.median(r.terminal_net_worth) for r in res.values()]
    p5 = [np.percentile(r.terminal_net_worth, 5) for r in res.values()]
    ce = {g: [crra_certainty_equivalent(r.terminal_net_worth, g) for r in res.values()] for g in (1, 3, 8)}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(alphas, np.array(med) / 1e3, "o-", label="Median")
    ax.plot(alphas, np.array(p5) / 1e3, "s--", label="5th percentile")
    for g, v in ce.items():
        ax.plot(alphas, np.array(v) / 1e3, ":", label=f"Certainty equiv. (γ={g})")
    ax.set(xlabel="α = share of discretionary cash sent to extra prepayment",
           ylabel="$k after 20 years", title="Effect of prepayment share α")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "alpha_sweep.png", dpi=150)


def fig_breakeven_heatmap():
    rates = np.linspace(0.03, 0.10, 8)
    mus = np.linspace(0.03, 0.11, 9)
    grid = np.empty((len(rates), len(mus)))
    for i, r in enumerate(rates):
        for j, mu in enumerate(mus):
            scn = base_scenario(loan=Loan(LOAN.principal, r, LOAN.term_months), mu=mu)
            res = run_all(scn, [INVEST_FIRST, PAYOFF_FIRST], 4000, seed=0)
            grid[i, j] = prob_a_beats_b(res[INVEST_FIRST.name].terminal_net_worth,
                                        res[PAYOFF_FIRST.name].terminal_net_worth)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(grid, origin="lower", cmap="RdBu", vmin=0, vmax=1, aspect="auto",
                   extent=[mus[0], mus[-1], rates[0], rates[-1]])
    ax.contour(mus, rates, grid, levels=[0.5], colors="k", linewidths=1.2)
    fig.colorbar(im, label="P(invest-first beats payoff-first)")
    ax.set(xlabel="Market drift μ", ylabel="Loan APR", title=f"Breakeven map (σ={SIGMA}, 20y horizon)")
    fig.tight_layout()
    fig.savefig(FIG / "breakeven_heatmap.png", dpi=150)


if __name__ == "__main__":
    res = table()
    fig_histogram(res)
    fig_alpha_sweep()
    fig_breakeven_heatmap()
    print(f"\nFigures written to {FIG}")
