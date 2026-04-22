"""
Quantum Approximate Optimization Algorithm (QAOA) for single ETF selection.
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

    def build_cost_hamiltonian(self, expected_returns, risk_penalty=0.0):
        """Build linear coefficients: negative expected returns."""
        return -expected_returns

    def qaoa_circuit(self, params, linear_coeffs, penalty, K):
        """QAOA circuit for selecting exactly one asset."""
        gamma = params[:self.n_layers]
        beta = params[self.n_layers:]

        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

        for layer in range(self.n_layers):
            # Cost Hamiltonian: RZ rotations based on linear coefficients
            for i in range(self.n_qubits):
                qml.RZ(2 * gamma[layer] * linear_coeffs[i], wires=i)
            # Mixer Hamiltonian: RX rotations
            for i in range(self.n_qubits):
                qml.RX(2 * beta[layer], wires=i)

        return qml.sample(wires=range(self.n_qubits))

    def compute_expectation(self, params, linear_coeffs, penalty, K):
        """Compute the expectation value of the cost Hamiltonian."""
        qnode = qml.QNode(self.qaoa_circuit, self.dev)
        samples = qnode(params, linear_coeffs, penalty, K)
        cost = 0.0
        for sample in samples:
            x = sample.astype(float)
            c = np.dot(linear_coeffs, x)
            # Penalty for not having exactly one asset selected
            c += penalty * (np.sum(x) - K) ** 2
            cost += c
        return cost / len(samples)

    def optimize_portfolio(self, expected_returns, covariance=None, risk_factor=0.5, penalty=100.0, K=1):
        """Run QAOA to select the single best ETF."""
        n = len(expected_returns)
        self.n_qubits = n
        linear_coeffs = self.build_cost_hamiltonian(expected_returns)

        init_params = np.random.uniform(0, np.pi, 2 * self.n_layers)

        def objective(params):
            return self.compute_expectation(params, linear_coeffs, penalty, K)

        result = minimize(objective, init_params, method='COBYLA', options={'maxiter': 100})
        optimal_params = result.x

        qnode = qml.QNode(self.qaoa_circuit, self.dev)
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
