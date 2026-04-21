"""
Main training script for Q-Quant engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from qaoa_optimizer import QAOAOptimizer
import push_results

def run_q_quant():
    print(f"=== P2-ETF-Q-QUANT Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]
        expected_returns = recent_returns.mean().values * 252  # Annualized
        covariance = recent_returns.cov().values * 252

        # Normalize expected returns for QUBO scaling
        scaler = StandardScaler()
        expected_returns_scaled = scaler.fit_transform(expected_returns.reshape(-1, 1)).flatten()

        # Initialize QAOA optimizer with number of assets
        n_assets = len(tickers)
        optimizer = QAOAOptimizer(
            n_qubits=n_assets,
            n_layers=config.QAOA_LAYERS,
            num_shots=config.NUM_SHOTS
        )

        print(f"  Running QAOA for {n_assets} assets...")
        best_bitstring, best_cost = optimizer.optimize_portfolio(
            expected_returns_scaled, covariance, risk_factor=0.5
        )

        # Decode bitstring to get selected tickers
        selected_tickers = [tickers[i] for i, bit in enumerate(best_bitstring) if bit == 1]
        print(f"  Selected tickers: {selected_tickers}")

        # Calculate portfolio metrics
        selected_indices = [i for i, bit in enumerate(best_bitstring) if bit == 1]
        if selected_indices:
            selected_returns = recent_returns.iloc[:, selected_indices]
            # Equal weight for simplicity
            weights = np.ones(len(selected_indices)) / len(selected_indices)
            portfolio_return = np.dot(selected_returns.mean().values * 252, weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(selected_returns.cov().values * 252, weights)))

            universe_result = {
                'selected_tickers': selected_tickers,
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'qaao_cost': best_cost,
                'bitstring': best_bitstring.tolist()
            }
        else:
            universe_result = {
                'selected_tickers': [],
                'portfolio_return': 0.0,
                'portfolio_risk': 0.0,
                'qaao_cost': best_cost,
                'bitstring': best_bitstring.tolist()
            }

        all_results[universe_name] = universe_result
        top_picks[universe_name] = selected_tickers

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "num_assets_to_select": config.NUM_ASSETS_TO_SELECT,
            "qaoa_layers": config.QAOA_LAYERS,
            "num_shots": config.NUM_SHOTS
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_q_quant()
