"""
Quantum Approximate Optimization Algorithm (QAOA) for selecting highest expected return ETF.
"""

import numpy as np
import pennylane as qml
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

class QAOAOptimizer:
    def __init__(self, n_qubits, n_layers=2, num_shots=1024):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_shots = num_shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=num_shots)

    def build_cost_hamiltonian(self, expected_returns):
        """Linear coefficients: negative expected returns (minimize cost = maximize return)."""
        return -expected_returns

    def qaoa_circuit(self, params, linear_coeffs):
        """QAOA circuit with only cost Hamiltonian (no mixer for simplicity)."""
        gamma = params[:self.n_layers]

        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(2 * gamma[layer] * linear_coeffs[i], wires=i)

        return qml.sample(wires=range(self.n_qubits))

    def compute_expectation(self, params, linear_coeffs):
        """Compute the expectation value of the cost Hamiltonian."""
        qnode = qml.QNode(self.qaoa_circuit, self.dev)
        samples = qnode(params, linear_coeffs)
        cost = 0.0
        for sample in samples:
            x = sample.astype(float)
            c = np.dot(linear_coeffs, x)
            cost += c
        return cost / len(samples)

    def optimize_portfolio(self, expected_returns, K=1):
        """Run QAOA to find the asset with highest expected return."""
        n = len(expected_returns)
        self.n_qubits = n
        linear_coeffs = self.build_cost_hamiltonian(expected_returns)

        init_params = np.random.uniform(0, np.pi, self.n_layers)

        def objective(params):
            return self.compute_expectation(params, linear_coeffs)

        result = minimize(objective, init_params, method='COBYLA', options={'maxiter': 100})
        optimal_params = result.x

        qnode = qml.QNode(self.qaoa_circuit, self.dev)
        samples = qnode(optimal_params, linear_coeffs)

        # Count frequency of each bitstring
        unique, counts = np.unique(samples, axis=0, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        top_bitstrings = unique[sorted_idx]

        # Convert bitstrings to ticker indices (the one with '1' selected)
        selected_indices = []
        for bs in top_bitstrings:
            idx = np.where(bs == 1)[0]
            if len(idx) == 1:
                selected_indices.append(idx[0])
            if len(selected_indices) >= 3:
                break

        return selected_indices
