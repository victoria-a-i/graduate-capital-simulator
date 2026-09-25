# Pay Down or Invest? A Stochastic Model of Post-Grad Capital Allocation

**AMS 325 group project, Stony Brook University**
Team: Victoria Irizarry · Irene Wu · Edward Kwong · *(open slots)*

## Research question
A new graduate has student debt *D* at APR *r* and a fixed monthly budget *B* for debt and investing.
What share **α** of the discretionary budget (*B* − minimum payment) should go to extra loan prepayment
instead of the stock market? How does the answer change with *r*, market drift μ, volatility σ, the time horizon,
and the graduate's risk aversion?

## Model (v0)
| Component | Implementation | File |
|---|---|---|
| Market | GBM from scratch, both the exact solution and Euler–Maruyama | `src/gradsim/gbm.py` |
| Loan | Fixed-rate monthly amortisation, prepayment allowed | `src/gradsim/debt.py` |
| Strategy | Fixed split α ∈ [0,1]; after payoff, the whole budget is invested | `src/gradsim/strategies.py` |
| Engine | Vectorised Monte Carlo with common random numbers across strategies | `src/gradsim/simulate.py` |
| Metrics | Percentiles, P(A beats B), CVaR, CRRA certainty equivalent, MC standard error | `src/gradsim/metrics.py` |
| Calibration | MLE of μ, σ from historical log returns (yfinance, optional) | `src/gradsim/calibrate.py` |

## Quick start
```bash
git clone <repo-url> && cd graduate-capital-simulator
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                         # 14 tests should pass
python scripts/run_baseline.py # prints a results table, writes figures/
```

## First results (placeholder parameters, not calibrated)
$30k loan at 6.5%, $800/month budget, 20-year horizon, μ = 8%, σ = 16%, 20,000 paths:

- Invest-first beats payoff-first on only **~51%** of paths, which is close to a coin flip.
- `figures/breakeven_heatmap.png`: the 50% line follows **r ≈ μ − σ²/2** (the *median* growth rate), not r ≈ μ.
  Volatility drag is the key mathematical result to explain in the report.
- `figures/alpha_sweep.png`: at these parameters, α shifts 20-year wealth by only about 1%. Next we need to find the
  regimes where the choice *matters*: large debt relative to the budget, short horizons, and high risk aversion.

## Roadmap
- [ ] **Calibrate** μ, σ from S&P 500 total-return data (^SP500TR). Report the standard error of μ, which is large.
- [ ] **Verification study**: weak and strong error of Euler vs. the exact scheme; convergence plots
- [ ] **Scenario grid**: debt size, APR, budget, horizon, σ, risk aversion γ
- [ ] **Extension 1**: Merton jump-diffusion (fat tails, crashes). Does the answer change?
- [ ] **Extension 2**: bootstrap historical returns as a non-parametric benchmark
- [ ] **Extension 3**: employer 401(k) match; "match-first" strategy
- [ ] **Extension 4 (stretch)**: income shocks / job loss; stochastic loan rates (Vasicek)
- [ ] **Out of scope, to state in the report**: taxes, income-driven repayment, forgiveness programs (check current federal rules at studentaid.gov)

## Team roles (draft, update on the progress slide)
| Role | Owner | Scope |
|---|---|---|
| Model & math | TBD | Derivations, volatility drag, jump-diffusion spec, report §2 |
| Simulation engine | TBD | `gbm.py`, `simulate.py`, performance, extensions |
| Data & verification | Victoria Irizarry | `calibrate.py`, convergence study, tests |
| Visualization & report | TBD | figures, slides, report assembly |

## Workflow
- Branch per feature: `git checkout -b feature/jump-diffusion`, then open a PR, and a teammate reviews it before merging.
- Everyone commits under their own account (the course checks individual contributions).
- `pytest` must pass before merging. Add a test for every new model component.
- Record AI assistance in `AI_LOG.md` (course requirement).

## Deliverables (per the course kickoff)
- [ ] 1-page proposal (title, members, summary, software), submitted on Brightspace
- [ ] One progress slide (goal, plan, roles)
- [ ] 5–10 min presentation (last class / final exam slot)
- [ ] ~4-page report: abstract & objectives; techniques & tools; observations & conclusions; contributions; references
- [ ] Code + data submitted through this repo

## References (starter; verify and extend)
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Higham, D. J. (2001). An algorithmic introduction to numerical simulation of SDEs. *SIAM Review*, 43(3).
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *J. Financial Economics*, 3.
- Kloeden, P. E. & Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
