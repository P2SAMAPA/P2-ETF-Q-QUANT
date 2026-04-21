"""
Quantum Approximate Optimization Algorithm (QAOA) for portfolio selection.
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

    def build_cost_hamiltonian(self, expected_returns, covariance, risk_factor=0.5, penalty=10.0):
        """Build the QUBO cost Hamiltonian coefficients."""
        n = len(expected_returns)
        # Linear terms: -expected_return * x_i
        linear_coeffs = -expected_returns
        # Quadratic terms: risk_factor * covariance * x_i * x_j
        quadratic_coeffs = risk_factor * covariance
        # Penalty for selecting exactly K assets
        # We'll add a penalty term: penalty * (sum(x_i) - K)^2
        K = self.n_qubits  # We're selecting all qubits? No, n_qubits is total assets.
        # For this implementation, we set n_qubits = number of assets.
        return linear_coeffs, quadratic_coeffs, penalty, K

    def qaoa_circuit(self, params, linear_coeffs, quadratic_coeffs, penalty, K):
        """QAOA circuit for portfolio optimization."""
        gamma = params[:self.n_layers]
        beta = params[self.n_layers:]

        # Initial state: uniform superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

        # QAOA layers
        for layer in range(self.n_layers):
            # Cost Hamiltonian
            # Linear terms: RZ rotations
            for i in range(self.n_qubits):
                qml.RZ(2 * gamma[layer] * linear_coeffs[i], wires=i)
            # Quadratic terms: ZZ interactions
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    if abs(quadratic_coeffs[i, j]) > 1e-6:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma[layer] * quadratic_coeffs[i, j], wires=j)
                        qml.CNOT(wires=[i, j])
            # Penalty for cardinality constraint: sum(x_i) - K
            # This requires a more complex implementation; simplified here.

            # Mixer Hamiltonian: RX rotations
            for i in range(self.n_qubits):
                qml.RX(2 * beta[layer], wires=i)

        # Measure all qubits in computational basis
        return qml.sample(wires=range(self.n_qubits))

    def compute_expectation(self, params, linear_coeffs, quadratic_coeffs, penalty, K):
        """Compute the expectation value of the cost Hamiltonian."""
        samples = self.qaoa_circuit(params, linear_coeffs, quadratic_coeffs, penalty, K)
        # Calculate cost for each sample
        cost = 0.0
        for sample in samples:
            x = sample  # binary vector
            # Linear contribution
            c = np.dot(linear_coeffs, x)
            # Quadratic contribution
            c += np.dot(x, np.dot(quadratic_coeffs, x))
            # Penalty for cardinality
            c += penalty * (np.sum(x) - K) ** 2
            cost += c
        return cost / len(samples)

    def optimize_portfolio(self, expected_returns, covariance, risk_factor=0.5):
        """Run QAOA to select optimal portfolio."""
        n = len(expected_returns)
        self.n_qubits = n
        linear_coeffs, quadratic_coeffs, penalty, K = self.build_cost_hamiltonian(
            expected_returns, covariance, risk_factor
        )

        # Initial parameters
        init_params = np.random.uniform(0, np.pi, 2 * self.n_layers)

        # Classical optimizer
        def objective(params):
            return self.compute_expectation(params, linear_coeffs, quadratic_coeffs, penalty, K)

        result = minimize(objective, init_params, method='COBYLA', options={'maxiter': 100})
        optimal_params = result.x

        # Sample from optimal circuit to get best bitstring
        samples = self.qaoa_circuit(optimal_params, linear_coeffs, quadratic_coeffs, penalty, K)
        # Find sample with minimum cost
        best_sample = None
        best_cost = float('inf')
        for sample in samples:
            x = sample
            c = np.dot(linear_coeffs, x) + np.dot(x, np.dot(quadratic_coeffs, x)) + penalty * (np.sum(x) - K) ** 2
            if c < best_cost:
                best_cost = c
                best_sample = x

        return best_sample, best_cost
