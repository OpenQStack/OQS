# visualize.py

import numpy as np
import matplotlib.pyplot as plt


def show_state(state, label="State", threshold=1e-3):
    """
    Pretty-print quantum state vector or density matrix.

    Args:
        state (np.ndarray): state vector or density matrix
        label (str): label for header
        threshold (float): minimum amplitude to display
    """
    print(f"\n🔎 {label}")
    dim = int(np.log2(state.shape[0])) if len(state.shape) == 2 else int(np.log2(len(state)))
    if len(state.shape) == 1:
        # Pure state vector
        for i, amp in enumerate(state):
            if np.abs(amp) > threshold:
                basis = format(i, f'0{dim}b')
                print(f"  |{basis}⟩: amplitude = {amp:.4f}")
    elif len(state.shape) == 2:
        # Density matrix
        print("Density matrix:")
        print(np.round(state, 4))
    else:
        raise ValueError("Unsupported state format")


def plot_probabilities(state, title="Measurement Probabilities"):
    """
    Plot probability distribution over computational basis states.

    Args:
        state (np.ndarray): state vector or density matrix
    """
    if len(state.shape) == 1:
        probs = np.abs(state) ** 2
    elif len(state.shape) == 2:
        probs = np.real(np.diag(state))
    else:
        raise ValueError("Unsupported state format")

    n = int(np.log2(len(probs)))
    labels = [format(i, f'0{n}b') for i in range(len(probs))]

    plt.figure(figsize=(6, 3))
    plt.bar(labels, probs)
    plt.title(title)
    plt.ylabel("Probability")
    plt.xlabel("Basis state")
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def bloch_vector(psi):
    """
    Compute Bloch vector for a single-qubit pure state |ψ⟩.

    Args:
        psi (np.ndarray): length-2 vector

    Returns:
        tuple: (x, y, z) components of Bloch vector
    """
    if len(psi) != 2:
        raise ValueError("bloch_vector requires a single-qubit state")

    a, b = psi
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(b) * a)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return np.real([x, y, z])


def plot_bloch(psi, title="Bloch Sphere"):
    """
    Plot a single-qubit state on the Bloch sphere.

    Args:
        psi (np.ndarray): single-qubit state vector
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)

    x, y, z = bloch_vector(psi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Draw sphere
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color='lightblue', alpha=0.2, linewidth=0)

    # Axes
    for axis, col in zip([(1, 0, 0), (0, 1, 0), (0, 0, 1)], ['r', 'g', 'b']):
        a = Arrow3D([0, axis[0]], [0, axis[1]], [0, axis[2]],
                    mutation_scale=15, lw=1.5, arrowstyle='-|>', color=col)
        ax.add_artist(a)

    # Bloch vector
    vec = Arrow3D([0, x], [0, y], [0, z],
                  mutation_scale=20, lw=2, arrowstyle='-|>', color='k')
    ax.add_artist(vec)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.show()
