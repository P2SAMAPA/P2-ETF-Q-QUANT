"""
Variational Quantum Eigensolver (VQE) for single ETF selection.
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

    def vqe_circuit(self, params, linear_coeffs, penalty, K):
        """Hardware-efficient ansatz."""
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

    def compute_expectation(self, params, linear_coeffs, penalty, K):
        qnode = qml.QNode(self.vqe_circuit, self.dev)
        samples = qnode(params, linear_coeffs, penalty, K)
        cost = 0.0
        for sample in samples:
            x = sample.astype(float)
            c = np.dot(linear_coeffs, x) + penalty * (np.sum(x) - K) ** 2
            cost += c
        return cost / len(samples)

    def optimize_portfolio(self, expected_returns, covariance=None, risk_factor=0.5, penalty=100.0, K=1):
        n = len(expected_returns)
        self.n_qubits = n
        linear_coeffs = self.build_cost_hamiltonian(expected_returns)

        init_params = np.random.uniform(0, np.pi, self.n_layers * self.n_qubits * 3)

        def objective(params):
            return self.compute_expectation(params, linear_coeffs, penalty, K)

        result = minimize(objective, init_params, method='COBYLA', options={'maxiter': 100})
        optimal_params = result.x

        qnode = qml.QNode(self.vqe_circuit, self.dev)
        samples = qnode(optimal_params, linear_coeffs, penalty, K)
        best_sample = None
        best_cost = float('inf')
        for sample in samples:
            x = sample.astype(float)
            c = np.dot(linear_coeffs, x) + penalty * (np.sum(x) - K) ** 2
            if c < best_cost:
                best_cost = c
                best_sample = x

        return best_sample, best_cost
