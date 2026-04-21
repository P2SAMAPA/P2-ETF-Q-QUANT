"""
Main training script for Q-Quant engine.
Runs QAOA and VQE in parallel for all universes.
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
    Returns (universe_name, result_dict, selected_tickers).
    """
    print(f"  [{optimizer_type}] Processing {universe_name}...")
    returns = returns_data[universe_name]
    if len(returns) < config.MIN_OBSERVATIONS:
        return universe_name, None, []

    recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]
    expected_returns = recent_returns.mean().values * 252
    covariance = recent_returns.cov().values * 252

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
        expected_returns_scaled, covariance, risk_factor=0.5
    )

    selected_tickers = [tickers[i] for i, bit in enumerate(best_bitstring) if bit == 1]
    print(f"  [{optimizer_type}] {universe_name} selected: {selected_tickers}")

    selected_indices = [i for i, bit in enumerate(best_bitstring) if bit == 1]
    if selected_indices:
        selected_returns = recent_returns.iloc[:, selected_indices]
        weights = np.ones(len(selected_indices)) / len(selected_indices)
        portfolio_return = np.dot(selected_returns.mean().values * 252, weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(selected_returns.cov().values * 252, weights)))

        result = {
            'selected_tickers': selected_tickers,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'qaao_cost' if optimizer_type == "QAOA" else 'vqe_cost': best_cost,
            'bitstring': best_bitstring.tolist()
        }
    else:
        result = {
            'selected_tickers': [],
            'portfolio_return': 0.0,
            'portfolio_risk': 0.0,
            'qaao_cost' if optimizer_type == "QAOA" else 'vqe_cost': best_cost,
            'bitstring': best_bitstring.tolist()
        }

    return universe_name, result, selected_tickers


def run_q_quant():
    print(f"=== P2-ETF-Q-QUANT Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    # Pre-compute returns for all universes to avoid data reload in subprocesses
    returns_data = {}
    tickers_map = {}
    for universe_name, tickers in config.UNIVERSES.items():
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) >= config.MIN_OBSERVATIONS:
            returns_data[universe_name] = returns
            tickers_map[universe_name] = tickers

    # Submit QAOA and VQE tasks for all universes in parallel
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

        # Collect results
        qaoa_results = {}
        qaoa_top_picks = {}
        vqe_results = {}
        vqe_top_picks = {}

        for future in as_completed(tasks):
            universe_name, result, selected = future.result()
            if result is None:
                continue
            # Determine which optimizer this came from by checking keys
            if 'qaao_cost' in result:
                qaoa_results[universe_name] = result
                qaoa_top_picks[universe_name] = selected
            else:
                vqe_results[universe_name] = result
                vqe_top_picks[universe_name] = selected

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "num_assets_to_select": config.NUM_ASSETS_TO_SELECT,
            "qaoa_layers": config.QAOA_LAYERS,
            "num_shots": config.NUM_SHOTS
        },
        "qaoa": {
            "universes": qaoa_results,
            "top_picks": qaoa_top_picks
        },
        "vqe": {
            "universes": vqe_results,
            "top_picks": vqe_top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_q_quant()
