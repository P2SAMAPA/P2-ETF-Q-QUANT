"""
Variational Quantum Eigensolver (VQE) for selecting highest expected return ETF.
"""

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

class VQEOptimizer:
    def __init__(self, n_qubits, n_layers=2, num_shots=1024):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_shots = num_shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=num_shots)

    def build_cost_hamiltonian(self, expected_returns):
        return -expected_returns

    def vqe_circuit(self, params, linear_coeffs):
        params = params.reshape(self.n_layers, self.n_qubits, 3)
        for i in range(self.n_qubits):
            qml.RY(params[0, i, 0], wires=i)
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i+1) % self.n_qubits])
            for i in range(self.n_qubits):
                qml.RY(params[layer, i, 1], wires=i)
                qml.RZ(params[layer, i, 2], wires=i)
        return qml.sample(wires=range(self.n_qubits))

    def compute_expectation(self, params, linear_coeffs):
        qnode = qml.QNode(self.vqe_circuit, self.dev)
        samples = qnode(params, linear_coeffs)
        cost = 0.0
        for sample in samples:
            x = sample.astype(float)
            cost += np.dot(linear_coeffs, x)
        return cost / len(samples)

    def optimize_portfolio(self, expected_returns, max_retries=3):
        n = len(expected_returns)
        self.n_qubits = n
        linear_coeffs = self.build_cost_hamiltonian(expected_returns)

        for attempt in range(max_retries):
            init_params = np.random.uniform(0, 2 * np.pi, self.n_layers * self.n_qubits * 3)

            def objective(params):
                return self.compute_expectation(params, linear_coeffs)

            result = minimize(objective, init_params, method='COBYLA',
                              options={'maxiter': 200, 'rhobeg': 0.5})
            optimal_params = result.x

            qnode = qml.QNode(self.vqe_circuit, self.dev)
            samples = qnode(optimal_params, linear_coeffs)

            unique, counts = np.unique(samples, axis=0, return_counts=True)
            sorted_idx = np.argsort(counts)[::-1]
            top_bitstrings = unique[sorted_idx]

            selected_indices = []
            for bs in top_bitstrings:
                idx = np.where(bs == 1)[0]
                if len(idx) == 1:
                    selected_indices.append(idx[0])
                if len(selected_indices) >= 3:
                    break

            if selected_indices:
                return selected_indices

        # Fallback
        fallback_idx = np.argsort(expected_returns)[::-1][:3]
        return list(fallback_idx)
