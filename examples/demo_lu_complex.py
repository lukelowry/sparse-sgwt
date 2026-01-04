# -*- coding: utf-8 -*-
"""
Example: LU Convolution with Complex Poles
------------------------------------------
Compares the LU-based solver (supporting complex poles) against the 
analytical Cholesky-based solver.
"""

import os, time, numpy as np, matplotlib.pyplot as plt
from sgwt import Convolve, impulse
from sgwt.lu_convolve import LUConvolve
from sgwt.util import VFKernel
from sgwt import DELAY_EASTWEST as L, COORD_EASTWEST as C
from demo_plot import plot_signal

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})

# 1. Setup signal: an impulse on the graph
X = impulse(L, n=65000)
SCALES = [0.1, 1.0, 10.0]

def f(x): 
    """Target analytical bandpass filter."""
    return np.stack([((4.0/s)*x / (x + 1.0/s)**2)**2 for s in SCALES], axis=1)

def vector_fitting(x, Y, n_poles=8, iterations=10):
    """
    Robust Vector Fitting algorithm to fit multiple real-valued responses Y to a
    common set of poles, ensuring the approximation is also real-valued.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (eigenvalues), shape (n_samples,).
    Y : np.ndarray
        Target responses, shape (n_samples, n_dims).
    n_poles : int
        Number of poles to use (must be even).
    iterations : int
        Number of iterations for pole refinement.

    Returns
    -------
    tuple
        (q, R, D) where q=-poles, R are residues, D is direct term.
    """
    n_samples, n_dims = Y.shape
    x = x.reshape(-1, 1)  # Ensure x is a column vector
    
    # 1. Initialize poles in LHP, as logarithmically spaced complex conjugate pairs.
    if n_poles % 2 != 0:
        raise ValueError("n_poles should be an even number for conjugate pairs.")
    
    n_half = n_poles // 2
    pole_freqs = np.geomspace(max(x.min(), 1e-5), x.max(), n_half)
    poles = np.concatenate([-pole_freqs + 1j * pole_freqs, -pole_freqs - 1j * pole_freqs])
 
    for iter_num in range(iterations):
        # 2. Build a REAL-VALUED linear system to enforce conjugate symmetry.
        # For each pole pair (p, p*), create two real basis functions from B=1/(x-p):
        #   - 2*Re(B) for the real part of the residue/coefficient
        #   - -2*Im(B) for the imaginary part of the residue/coefficient
        poles = poles[np.argsort(poles.imag)]
        pos_imag_poles = poles[n_half:]

        basis_real = np.zeros((n_samples, n_half))
        basis_imag = np.zeros((n_samples, n_half))
        for i in range(n_half):
            b_i = 1.0 / (x - pos_imag_poles[i])
            basis_real[:, i] = 2 * b_i.real.flatten()
            basis_imag[:, i] = -2 * b_i.imag.flatten()

        # A_res_block corresponds to unknowns [Re(R_d), Im(R_d), D_d]
        A_res_block = np.c_[basis_real, basis_imag, np.ones(n_samples)]
        A_res = np.kron(np.eye(n_dims), A_res_block)
        
        # A_scale corresponds to unknowns [Re(c), Im(c)]
        A_scale_list = [-Y[:, d, None] * np.c_[basis_real, basis_imag] for d in range(n_dims)]
        A_scale = np.vstack(A_scale_list)
        
        A = np.hstack([A_res, A_scale])
        b = Y.flatten(order='F')
        
        # 3. Solve the real-valued LLS problem
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # 4. Relocate poles by finding the zeros of the scaling function sigma(s).
        # First, reconstruct complex scaling coefficients 'c' from the real solution.
        sol_c = sol[n_dims * (n_poles + 1):]
        c = np.zeros(n_poles, dtype=complex)
        c[n_half:] = sol_c[:n_half] + 1j * sol_c[n_half:]  # For positive imag poles
        c[:n_half] = np.conj(c[n_half:][::-1])             # For negative imag poles

        # New poles are eigenvalues of (diag(p) - 1 * c^T)
        poles = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(n_poles), c))
        
        # 5. Enforce conjugate pairs and stability for a real-valued filter response.
        poles = poles[np.argsort(poles.imag)]
        for i in range(n_half):
            avg_pole = (poles[n_half + i] + np.conj(poles[n_half - 1 - i])) / 2.0
            poles[n_half + i] = avg_pole
            poles[n_half - 1 - i] = np.conj(avg_pole)

        unstable_idx = np.where(poles.real > 0)[0]
        poles[unstable_idx] -= 2 * poles[unstable_idx].real
 
    # 6. Final LLS fit for residues [R] and direct term [D] with the final, stable poles.
    # This must also be a real-valued LLS to enforce conjugate residues.
    A_final_block = np.c_[basis_real, basis_imag, np.ones(n_samples)]
    A_final = np.kron(np.eye(n_dims), A_final_block)
    
    sol_final, _, _, _ = np.linalg.lstsq(A_final, b, rcond=None)
    
    # Reshape solution and reconstruct complex residues R
    sol_reshaped = sol_final.reshape((n_dims, n_poles + 1)).T
    R_real = sol_reshaped[:n_half, :]
    R_imag = sol_reshaped[n_half:n_poles, :]
    D = sol_reshaped[-1, :].copy()
    
    R = np.zeros((n_poles, n_dims), dtype=complex)
    R[n_half:] = R_real + 1j * R_imag
    R[:n_half] = np.conj(R[n_half:][::-1])
        
    # Return solver poles (q = -p), residues, and direct term
    return -poles, R, D

# 2. Generate kernel via Vector Fitting
print("Fitting complex poles to target filters...")
XMIN = 1e-4
XMAX = 1e4
x_fit = np.geomspace(XMIN, XMAX, 500)
y_fit = f(x_fit)
q, r, d = vector_fitting(x_fit, y_fit, n_poles=6)

# Create VFKernel object
K_vf = VFKernel(R=r, Q=q, D=d)

# 3. Calculations
print(f"Benchmarking LU vs Cholesky on {L.shape[0]} nodes...")

# Analytical Baseline (Cholesky-based)
with Convolve(L) as conv:
    start = time.time()
    Y_analytical = conv.bandpass(X, SCALES, order=2)
    t_chol = time.time() - start

# LU Convolution with Complex Poles
with LUConvolve(L) as conv_lu:
    start = time.time()
    Y_lu = conv_lu.convolve(X, K_vf)
    t_lu = time.time() - start

# --- Plotting (Style adapted from demo_cheby_convolve_1.py) ---
fig = plt.figure(figsize=(12, 8))
gs_main = fig.add_gridspec(1, 2, width_ratios=[0.8, 2], wspace=0.05)
fig.suptitle("LU Convolution with Complex Poles", fontsize=16, fontweight='bold')
fig.text(0.5, 0.935, "Comparing Analytical (Real) vs. LU (Complex Pole Expansion)", ha='center', fontsize=12, style='italic')

# --- Left Column: Analysis Plots ---
gs_left = gs_main[0, 0].subgridspec(3, 1, hspace=0.5)

# Spectral plot (Analytical Target vs Complex Approximation)
ax_spec = fig.add_subplot(gs_left[0, 0])
x_eval = np.geomspace(XMIN, XMAX, 1000)
y_true = f(x_eval)

# Evaluate complex kernel response: H(lambda) = sum( r / (lambda + q) ) + d
y_complex = np.zeros((len(x_eval), len(SCALES)), dtype=complex) + K_vf.D
for i in range(len(K_vf.Q)):
    qi = K_vf.Q[i]
    ri = K_vf.R[i, :]
    y_complex += ri / (x_eval[:, None] + qi)

for i in range(len(SCALES)):
    line, = ax_spec.plot(x_eval, y_true[:, i], '--', alpha=0.3, label=f'Target s={SCALES[i]}')
    color = line.get_color()
    ax_spec.plot(x_eval, y_complex[:, i].real, '-', color=color, label=f'Re(H) s={SCALES[i]}')
    ax_spec.plot(x_eval, y_complex[:, i].imag, ':', color=color, alpha=0.6) # Imaginary part

ax_spec.set_xscale('log')
ax_spec.set_title("Spectral Response Comparison", fontsize=12, fontweight='bold', pad=10)
ax_spec.grid(True, alpha=0.3, linestyle='--')
ax_spec.spines[['top', 'right']].set_visible(False)
ax_spec.legend(fontsize=7, ncol=2, loc='upper left')

# Timing plot
ax_time = fig.add_subplot(gs_left[1, 0])
labels = ['Cholesky', 'LU (Complex)']
times_ms = [t_chol * 1000, t_lu * 1000]
bars = ax_time.bar(labels, times_ms, color=['#66c2a5', '#fc8d62'], width=0.4)
ax_time.set_ylabel('Runtime [ms]', fontsize=10)
ax_time.set_title('Convolution Runtime', fontsize=12, fontweight='bold', pad=10)
ax_time.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    ax_time.annotate(f"{yval:.1f}", xy=(bar.get_x() + bar.get_width()/2, yval),
                     xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)
ax_time.spines[['top', 'right']].set_visible(False)

# Info plot
ax_info = fig.add_subplot(gs_left[2, 0])
ax_info.axis('off')
ax_info.text(0, 1, f"Nodes: {L.shape[0]}\nComplex Poles: {len(K_vf.Q)}\n"
                   f"Max Im(Y_lu): {np.max(np.abs(Y_lu.imag)):.2e}\n\n"
                   "LU decomposition allows for\ncomplex shifts (L + qI),\n"
                   "enabling resonant filter\napproximations via VF.", 
             transform=ax_info.transAxes, va='top', fontsize=9, family='monospace')

# --- Right Column: Spatial Wavelets ---
gs_right = gs_main[0, 1].subgridspec(len(SCALES), 2, hspace=0.0, wspace=0.0)

for i, s in enumerate(SCALES):
    # Analytical (Cholesky)
    ax_a = fig.add_subplot(gs_right[i, 0])
    plot_signal(Y_analytical[i][:, 0], C, 'coolwarm', ax=ax_a)
    if i == 0: ax_a.set_title("Analytical (Cholesky)", fontsize=12, fontweight='bold', pad=10)

    # LU (Complex Poles)
    ax_l = fig.add_subplot(gs_right[i, 1])
    plot_signal(Y_lu[:, 0, i].real, C, 'coolwarm', ax=ax_l)
    if i == 0: ax_l.set_title("Complex Pole (LU)", fontsize=12, fontweight='bold', pad=10)
    ax_l.text(1.02, 0.5, f"Scale {s}", transform=ax_l.transAxes, rotation=-90, va='center', fontweight='bold')

plt.tight_layout()
img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', '_static', 'images', 'demo_lu_complex.png'))
os.makedirs(os.path.dirname(img_path), exist_ok=True)
plt.savefig(img_path, dpi=400, bbox_inches='tight')
plt.show()