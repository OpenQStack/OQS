# qec.py
"""
QEC Core Module (openqstack/qec.py)

Contains:
- BitFlipCode: 3-qubit quantum error correction demo
- ErrorChannel base class and standard noise models
- tensor utilities for operator composition

Designed for educational, simulation, and prototype control use.
"""

import numpy as np
import random

class BitFlipCode:
    """
    3-qubit bit-flip code.
    Encodes 1 logical qubit to 3 physical qubits.
    Corrects a single bit-flip (X) error.
    """
    def __init__(self):
        self.ket0 = np.array([1, 0])
        self.ket1 = np.array([0, 1])
        self.I = np.eye(2)
        self.X = np.array([[0, 1], [1, 0]])

    def encode(self, psi):
        """
        Encode a 1-qubit state |ψ⟩ = α|0⟩ + β|1⟩ into 3-qubit bit-flip code.
        """
        alpha, beta = psi
        psi0 = np.kron(self.ket0, np.kron(self.ket0, self.ket0))
        psi1 = np.kron(self.ket1, np.kron(self.ket1, self.ket1))
        return alpha * psi0 + beta * psi1

    def apply_random_X_error(self, state):
        """
        Applies a single bit-flip (Pauli X) error to a random qubit.
        """
        qubit = random.choice([0, 1, 2])
        ops = [self.I, self.I, self.I]
        ops[qubit] = self.X
        error_op = tensor(*ops)

    def measure_syndrome(self, state):
        """
        Simulate measurement of parity checks.
        For simplicity: infer syndrome by inspecting the state.
        """
        z_basis = [np.kron(b1, np.kron(b2, b3))
                   for b1 in [self.ket0, self.ket1]
                   for b2 in [self.ket0, self.ket1]
                   for b3 in [self.ket0, self.ket1]]
        probs = np.abs([np.vdot(b, state) for b in z_basis])
        i_max = np.argmax(probs)
        bits = format(i_max, '03b')
        return bits

    def recover(self, state, syndrome):
        """
        Majority vote decoder: flips minority bit.
        """
        bit_counts = [int(b) for b in syndrome]
        if bit_counts.count(1) == 2:
            flip_index = bit_counts.index(0)
        elif bit_counts.count(0) == 2:
            flip_index = bit_counts.index(1)
        else:
            return state  # No correction needed

        ops = [self.I, self.I, self.I]
        ops[flip_index] = self.X
        recovery_op = self._tensor3(*ops)
        return recovery_op @ state

    def decode(self, state):
        """
        Project back to logical qubit by collapsing to |000⟩ or |111⟩.
        """
        psi0 = np.kron(self.ket0, np.kron(self.ket0, self.ket0))
        psi1 = np.kron(self.ket1, np.kron(self.ket1, self.ket1))
        amp0 = np.vdot(psi0, state)
        amp1 = np.vdot(psi1, state)
        norm = np.sqrt(abs(amp0)**2 + abs(amp1)**2)
        return np.array([amp0, amp1]) / norm

    def _tensor3(self, A, B, C):
        return np.kron(A, np.kron(B, C))


# --- FUTURE-READY STRUCTURES ---

import numpy as np
import random

class ErrorChannel:
    """
    Abstract base class for quantum noise channels.

    Provides interface for:
    - Defining Kraus operators
    - Applying noise stochastically or deterministically
    - Extensibility for custom noise, diagnostics, batch evaluation
    """

    def __init__(self, n_qubits=1, seed=None):
        """
        Args:
            n_qubits (int): number of qubits this channel acts on.
            seed (int or None): optional seed for reproducibility.
        """
        self.n_qubits = n_qubits
        self.rng = random.Random(seed)

    def kraus_operators(self):
        """
        Return a list of (K_i, p_i) tuples representing Kraus operators and their probabilities.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement kraus_operators.")

    def apply(self, state, mode="stochastic"):
        """
        Apply the channel to a quantum state.

        Args:
            state (np.ndarray): pure or mixed state (state vector or density matrix)
            mode (str): 'stochastic' or 'average' (deterministic Kraus sum)

        Returns:
            np.ndarray: new state after noise is applied.
        """
        kraus_ops = self.kraus_operators()

        if mode == "stochastic":
            ops, probs = zip(*kraus_ops)
            chosen_op = self.rng.choices(ops, weights=probs, k=1)[0]
            return chosen_op @ state

        elif mode == "average":
            if len(state.shape) == 1:  # pure state → density matrix
                state = np.outer(state, np.conj(state))
            rho_out = np.zeros_like(state, dtype=complex)
            for K, _ in kraus_ops:
                rho_out += K @ state @ K.conj().T
            return rho_out

        else:
            raise ValueError(f"Unknown mode '{mode}': use 'stochastic' or 'average'.")

    def __call__(self, state, **kwargs):
        """Allows calling channel as a function."""
        return self.apply(state, **kwargs)

    def description(self):
        """Optional: Return a string description of the channel for logging/metadata."""
        return f"{self.__class__.__name__} acting on {self.n_qubits} qubit(s)"

# --- Common Channel Implementations ---

class BitFlipChannel(ErrorChannel):
    def __init__(self, p, n_qubits=1, target_qubit=0):
        super().__init__(n_qubits)
        self.p = p
        self.target = target_qubit

    def kraus_operators(self):
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        K0 = np.sqrt(1 - self.p) * I
        K1 = np.sqrt(self.p) * X
        return [
            (self._expand(K0), 1 - self.p),
            (self._expand(K1), self.p)
        ]

    def _expand(self, op):
        """Expand single-qubit op to full n-qubit space."""
        ops = [np.eye(2)] * self.n_qubits
        ops[self.target] = op
        return tensor(*ops)


class PhaseFlipChannel(ErrorChannel):
    def __init__(self, p, n_qubits=1, target_qubit=0):
        super().__init__(n_qubits)
        self.p = p
        self.target = target_qubit

    def kraus_operators(self):
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        K0 = np.sqrt(1 - self.p) * I
        K1 = np.sqrt(self.p) * Z
        return [
            (self._expand(K0), 1 - self.p),
            (self._expand(K1), self.p)
        ]

    def _expand(self, op):
        ops = [np.eye(2)] * self.n_qubits
        ops[self.target] = op
        return tensor(*ops)


class DepolarizingChannel(ErrorChannel):
    def __init__(self, p, n_qubits=1, target_qubit=0):
        super().__init__(n_qubits)
        self.p = p
        self.target = target_qubit

    def kraus_operators(self):
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        p0 = 1 - self.p
        p_rest = self.p / 3

        return [
            (self._expand(np.sqrt(p0) * I), p0),
            (self._expand(np.sqrt(p_rest) * X), p_rest),
            (self._expand(np.sqrt(p_rest) * Y), p_rest),
            (self._expand(np.sqrt(p_rest) * Z), p_rest),
        ]

    def _expand(self, op):
        ops = [np.eye(2)] * self.n_qubits
        ops[self.target] = op
        return tensor(*ops)


# --- Utility ---

import numpy as np

def tensor(*ops):
    """
    Computes the k-fold tensor (Kronecker) product of input operators.

    Args:
        *ops: Sequence of operators (numpy arrays, ints, bools, or symbolic matrices).
              Each must be square matrices or valid scalars.

    Returns:
        numpy.ndarray or symbolic matrix representing the total tensor product.

    Raises:
        ValueError: if input is empty or contains non-square matrices.
    """
    if not ops:
        raise ValueError("tensor() requires at least one input.")

    result = _to_matrix(ops[0])
    for op in ops[1:]:
        result = np.kron(result, _to_matrix(op))
    return result


def _to_matrix(op):
    """
    Convert input to a 2D matrix if needed.
    Accepts scalars (int, float, bool), vectors, or matrices.
    """
    if isinstance(op, (int, float, complex, bool)):
        return np.array([[op]], dtype=complex)

    op = np.asarray(op)

    if op.ndim == 1:
        # Treat vectors as column vectors
        return op[:, np.newaxis]
    elif op.ndim == 2:
        if op.shape[0] != op.shape[1]:
            raise ValueError(f"Operator must be square: got shape {op.shape}")
        return op
    else:
        raise ValueError(f"Invalid operator shape: {op.shape}")

