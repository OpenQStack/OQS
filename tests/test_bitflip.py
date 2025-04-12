
import numpy as np
from openqstack.qec import BitFlipCode

def test_no_error():
    psi = [1, 0]
    code = BitFlipCode()
    encoded = code.encode(psi)
    recovered = code.recover(encoded, "000")  # No error
    decoded = code.decode(recovered)
    assert np.allclose(decoded, psi, atol=1e-6)

def test_single_X_error_qubit_0():
    psi = [0, 1]
    code = BitFlipCode()
    encoded = code.encode(psi)
    corrupted = code.apply_random_X_error(encoded)  # force this in future
    syndrome = code.measure_syndrome(corrupted)
    recovered = code.recover(corrupted, syndrome)
    decoded = code.decode(recovered)
    assert np.allclose(np.abs(decoded), np.abs(psi), atol=1e-6)

def test_encoded_logical_states_are_normalized():
    psi = [1/np.sqrt(2), 1j/np.sqrt(2)]
    code = BitFlipCode()
    encoded = code.encode(psi)
    norm = np.linalg.norm(encoded)
    assert np.isclose(norm, 1.0, atol=1e-6)
