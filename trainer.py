"""
Main training script for Q-Quant engine.
Runs daily (252d) and global (2008‑present) training for QAOA and VQE.
"""

import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
import data_manager
from qaoa_optimizer import QAOAOptimizer
from vqe_optimizer import VQEOptimizer
import push_results

def process_universe(optimizer_type, universe_name, tickers, returns, mode="daily"):
    """Process a single universe with either QAOA or VQE."""
    print(f"  [{optimizer_type}][{mode}] Processing {universe_name}...")
    if len(returns) < config.MIN_OBSERVATIONS:
        return optimizer_type, mode, universe_name, None, None, []

    if mode == "daily":
        recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]
    else:
        recent_returns = returns

    # Use raw annualized expected returns (NO SCALING)
    expected_returns = recent_returns.mean().values * 252

    n_assets = len(tickers)
    if optimizer_type == "QAOA":
        optimizer = QAOAOptimizer(n_qubits=n_assets, n_layers=config.QAOA_LAYERS, num_shots=config.NUM_SHOTS)
    else:
        optimizer = VQEOptimizer(n_qubits=n_assets, n_layers=config.QAOA_LAYERS, num_shots=config.NUM_SHOTS)

    # Pass raw expected returns; penalty must be large enough relative to return magnitudes
    # Typical annualized returns are in [‑0.5, 1.5], so penalty=10.0 is sufficient
    best_bitstring, _ = optimizer.optimize_portfolio(expected_returns, penalty=10.0, K=1)

    selected_index = np.argmax(best_bitstring)
    selected_ticker = tickers[selected_index]
    selected_return = expected_returns[selected_index]

    # Top 3 by expected return
    top3 = []
    sorted_indices = np.argsort(expected_returns)[::-1][:3]
    for idx in sorted_indices:
        top3.append({"ticker": tickers[idx], "expected_return": expected_returns[idx]})

    print(f"  [{optimizer_type}][{mode}] {universe_name} selected: {selected_ticker} ({selected_return*100:.2f}%)")
    return optimizer_type, mode, universe_name, selected_ticker, selected_return, top3


def run_q_quant():
    print(f"=== P2-ETF-Q-QUANT Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.GLOBAL_TRAINING_START]

    returns_global = {}
    tickers_map = {}
    for universe_name, tickers in config.UNIVERSES.items():
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) >= config.MIN_OBSERVATIONS:
            returns_global[universe_name] = returns
            tickers_map[universe_name] = tickers

    tasks = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for opt in ["QAOA", "VQE"]:
            for uni in returns_global.keys():
                daily_returns = returns_global[uni].iloc[-config.LOOKBACK_WINDOW:]
                tasks.append(executor.submit(
                    process_universe, opt, uni, tickers_map[uni], daily_returns, "daily"
                ))
                tasks.append(executor.submit(
                    process_universe, opt, uni, tickers_map[uni], returns_global[uni], "global"
                ))

        results = {
            "daily": {"QAOA": {}, "VQE": {}},
            "global": {"QAOA": {}, "VQE": {}}
        }

        for future in as_completed(tasks):
            opt_type, mode, uni, ticker, exp_ret, top3 = future.result()
            if ticker:
                results[mode][opt_type][uni] = {
                    "ticker": ticker,
                    "expected_return": exp_ret,
                    "top3": top3
                }

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "global_training_start": config.GLOBAL_TRAINING_START,
            "num_assets_to_select": 1
        },
        "daily": results["daily"],
        "global": results["global"]
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_q_quant()
