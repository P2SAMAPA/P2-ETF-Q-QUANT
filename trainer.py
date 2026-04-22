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
    """
    Process a single universe with either QAOA or VQE.
    Returns (universe_name, selected_ticker, expected_return).
    """
    print(f"  [{optimizer_type}] Processing {universe_name}...")
    returns = returns_data[universe_name]
    if len(returns) < config.MIN_OBSERVATIONS:
        return universe_name, None, None

    recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]
    expected_returns = recent_returns.mean().values * 252  # Annualized

    scaler = StandardScaler()
    expected_returns_scaled = scaler.fit_transform(expected_returns.reshape(-1, 1)).flatten()

    n_assets = len(tickers)
    if optimizer_type == "QAOA":
        optimizer = QAOAOptimizer(
            n_qubits=n_assets,
            n_layers=config.QAOA_LAYERS,
            num_shots=config.NUM_SHOTS
        )
    else:
        optimizer = VQEOptimizer(
            n_qubits=n_assets,
            n_layers=config.QAOA_LAYERS,
            num_shots=config.NUM_SHOTS
        )

    best_bitstring, best_cost = optimizer.optimize_portfolio(
        expected_returns_scaled, penalty=100.0, K=1
    )

    # Find the selected asset (bit = 1)
    selected_index = np.argmax(best_bitstring)
    selected_ticker = tickers[selected_index]
    selected_return = expected_returns[selected_index]

    print(f"  [{optimizer_type}] {universe_name} selected: {selected_ticker} (return: {selected_return*100:.2f}%)")

    return universe_name, selected_ticker, selected_return


def run_q_quant():
    print(f"=== P2-ETF-Q-QUANT Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    # Pre-compute returns for all universes
    returns_data = {}
    tickers_map = {}
    for universe_name, tickers in config.UNIVERSES.items():
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) >= config.MIN_OBSERVATIONS:
            returns_data[universe_name] = returns
            tickers_map[universe_name] = tickers

    # Submit QAOA and VQE tasks in parallel
    tasks = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for optimizer_type in ["QAOA", "VQE"]:
            for universe_name in returns_data.keys():
                tasks.append(executor.submit(
                    process_universe_optimizer,
                    optimizer_type,
                    universe_name,
                    tickers_map[universe_name],
                    returns_data
                ))

        qaoa_top_picks = {}
        vqe_top_picks = {}

        for future in as_completed(tasks):
            universe_name, ticker, exp_return = future.result()
            if ticker is None:
                continue
            # Determine which optimizer this came from by checking the task string
            # We'll store in both; they'll be separated by the loop structure
            # Since we can't directly know, we'll use a simple heuristic: store in both for now
            # Better: modify process_universe_optimizer to return optimizer_type
            # Let's update the return signature
            pass

    # We need to adjust the return signature to include optimizer_type
    # I'll rewrite the parallel section clearly below
