
from openqstack.qec import BitFlipCode
from openqstack.visualize import show_state, plot_probabilities

# Define logical qubit |ψ⟩ = |0⟩
psi = [1, 0]

# Encode
code = BitFlipCode()
encoded = code.encode(psi)
print("🎯 Encoded logical qubit:")
show_state(encoded)

# Apply random bit-flip error
corrupted = code.apply_random_X_error(encoded)
print("⚡ Corrupted state after X error:")
show_state(corrupted)

# Measure syndrome
syndrome = code.measure_syndrome(corrupted)
print(f"🧪 Syndrome detected: {syndrome}")

# Recover
recovered = code.recover(corrupted, syndrome)
print("🛠️ Recovered encoded state:")
show_state(recovered)

# Decode
decoded = code.decode(recovered)
print("✅ Decoded logical qubit:")
show_state(decoded)
plot_probabilities(decoded, title="Final logical qubit probabilities")
