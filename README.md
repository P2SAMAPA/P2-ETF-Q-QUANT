# P2-ETF-Q-QUANT

**Quantum‑Classical Hybrid Engine for ETF Portfolio Selection with QAOA and VQE**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-Q-QUANT/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-Q-QUANT/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--q--quant--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-q-quant-results)

## Overview

`P2-ETF-Q-QUANT` formulates the ETF selection problem as a **QUBO** (Quadratic Unconstrained Binary Optimization) and solves it with two variational quantum algorithms:

- **QAOA** (Quantum Approximate Optimization Algorithm)
- **VQE** (Variational Quantum Eigensolver)

Both algorithms run **in parallel** on a classical simulator (PennyLane). The engine selects a portfolio of **5 ETFs** per universe that maximizes expected return while controlling risk, using quantum‑inspired tunneling to explore the combinatorial search space.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

Data source: [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

## Methodology

1. **QUBO Formulation**: Expected returns and covariance (from 252‑day rolling window) are encoded into a quadratic cost function with a cardinality penalty (`K = 5`).
2. **Quantum Optimization**:
   - **QAOA**: Alternating cost‑and‑mixer layers approximate the ground state.
   - **VQE**: Hardware‑efficient ansatz minimizes the energy of the QUBO Hamiltonian.
3. **Parallel Execution**: Both optimizers run simultaneously using `ProcessPoolExecutor`, reducing total runtime.
4. **Portfolio Construction**: Selected ETFs are equally weighted. Portfolio return, risk, and Sharpe ratio are reported.

## File Structure
P2-ETF-Q-QUANT/
├── config.py # Paths, universes, quantum parameters
├── data_manager.py # Data loading and preprocessing
├── qaoa_optimizer.py # QAOA implementation (PennyLane)
├── vqe_optimizer.py # VQE implementation (PennyLane)
├── trainer.py # Orchestrates parallel QAOA + VQE
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Dashboard with two main tabs (QAOA / VQE)
├── us_calendar.py # U.S. market calendar utilities
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LOOKBACK_WINDOW` | 252 | Days of historical data |
| `NUM_ASSETS_TO_SELECT` | 5 | Number of ETFs in portfolio |
| `QAOA_LAYERS` | 2 | Circuit depth (used by both algorithms) |
| `NUM_SHOTS` | 1024 | Measurement shots for sampling |

## Running Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-Q-QUANT.git
cd P2-ETF-Q-QUANT
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
Dashboard Features
Two Main Tabs: Switch between QAOA and VQE results.

Sub‑tabs per Universe: Combined, Equity Sectors, FI/Commodities.

Hero Cards: Selected portfolio with expected return, risk, and Sharpe ratio.

ETF List: All selected tickers for the universe.

Next Trading Day: U.S. market calendar integration.

Performance
Runtime: ~15‑30 minutes on GitHub Actions (parallel execution).

Simulator: PennyLane's default.qubit (CPU‑based, no GPU required).

License
MIT License
