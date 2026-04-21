# P2-ETF-Q-QUANT

**Quantum-Classical Hybrid Engine for ETF Portfolio Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-Q-QUANT/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-Q-QUANT/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--q--quant--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-q-quant-results)

## Overview

`P2-ETF-Q-QUANT` uses the **Quantum Approximate Optimization Algorithm (QAOA)** to solve the combinatorial problem of selecting the optimal subset of ETFs. The problem is formulated as a QUBO and solved with a quantum‑classical hybrid approach.

## Methodology

1. **QUBO Formulation**: Expected returns and covariance matrix are encoded into a QUBO problem.
2. **QAOA Optimization**: A parameterized quantum circuit is optimized to find the ground state (optimal asset selection).
3. **Portfolio Construction**: Selected ETFs are equally weighted.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)

## Usage
```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
