"""
Main training script for Q-Quant engine.
Selects exactly one ETF per universe using QAOA and VQE in parallel.
"""

import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler

import config
import data_manager
from qaoa_optimizer import QAOAOptimizer
from vqe_optimizer import VQEOptimizer
import push_results

def process_universe_optimizer(optimizer_type, universe_name, tickers, returns_data):
    """Process a single universe with either QAOA or VQE."""
    print(f"  [{optimizer_type}] Processing {universe_name}...")
    returns = returns_data[universe_name]
    if len(returns) < config.MIN_OBSERVATIONS:
        return optimizer_type, universe_name, None, None

    recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]
    expected_returns = recent_returns.mean().values * 252

    scaler = StandardScaler()
    expected_returns_scaled = scaler.fit_transform(expected_returns.reshape(-1, 1)).flatten()

    n_assets = len(tickers)
    if optimizer_type == "QAOA":
        optimizer = QAOAOptimizer(n_qubits=n_assets, n_layers=config.QAOA_LAYERS, num_shots=config.NUM_SHOTS)
    else:
        optimizer = VQEOptimizer(n_qubits=n_assets, n_layers=config.QAOA_LAYERS, num_shots=config.NUM_SHOTS)

    best_bitstring, _ = optimizer.optimize_portfolio(expected_returns_scaled, penalty=100.0, K=1)

    selected_index = np.argmax(best_bitstring)
    selected_ticker = tickers[selected_index]
    selected_return = expected_returns[selected_index]

    print(f"  [{optimizer_type}] {universe_name} selected: {selected_ticker} ({selected_return*100:.2f}%)")
    return optimizer_type, universe_name, selected_ticker, selected_return


def run_q_quant():
    print(f"=== P2-ETF-Q-QUANT Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    returns_data = {}
    tickers_map = {}
    for universe_name, tickers in config.UNIVERSES.items():
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) >= config.MIN_OBSERVATIONS:
            returns_data[universe_name] = returns
            tickers_map[universe_name] = tickers

    tasks = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for opt in ["QAOA", "VQE"]:
            for uni in returns_data.keys():
                tasks.append(executor.submit(
                    process_universe_optimizer, opt, uni, tickers_map[uni], returns_data
                ))

        qaoa_picks = {}
        vqe_picks = {}
        for future in as_completed(tasks):
            opt_type, uni, ticker, exp_ret = future.result()
            if ticker:
                if opt_type == "QAOA":
                    qaoa_picks[uni] = {"ticker": ticker, "expected_return": exp_ret}
                else:
                    vqe_picks[uni] = {"ticker": ticker, "expected_return": exp_ret}

    output_payload = {
        "run_date": config.TODAY,
        "config": {"lookback_window": config.LOOKBACK_WINDOW, "num_assets_to_select": 1},
        "qaoa": {"top_picks": qaoa_picks},
        "vqe": {"top_picks": vqe_picks}
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_q_quant()
