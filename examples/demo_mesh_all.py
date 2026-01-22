import numpy as np
from examples.demo_mesh_plot import plot_mesh_wavelet_custom_coords
import sgwt
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- 1. CORE PROCESSING FUNCTIONS ---

def get_saliency_signal(conv, coords, scale, percentile=75, power=1.5):
    """
    Computes a clean spectral saliency signal at a specific scale.
    - Isolates geometric features of a specific 'size'.
    - Suppresses the 'nominal' bunny to zero (white).
    """
    # 1. Compute Wavelet coefficients for the XYZ geometry
    # This returns a vector (dx, dy, dz) representing geometric variation
    wavelets = conv.bandpass(coords, scales=[scale])[0]
    
    # 2. Compute magnitude (L2 norm)
    sig = np.linalg.norm(wavelets, axis=1)
    
    # 3. Apply Noise Floor (Percentile Thresholding)
    floor = np.percentile(sig, percentile)
    sig = np.clip(sig - floor, 0, None)
    
    # 4. Non-linear enhancement (makes the peaks 'pop' more)
    sig = sig ** power
    
    # Normalize to [0, 1]
    if sig.max() > 0:
        sig /= sig.max()
    return sig

# --- 2. MAIN EXECUTION SCRIPT ---

def generate_bunny_variations():
    # Load assets from the SGWT library
    L = sgwt.MESH_BUNNY
    coords = np.asfortranarray(sgwt.BUNNY_XYZ)
    
    print("Initializing Spectral Engine...")
    
    # Use the Convolve context to factorize the Laplacian once
    with sgwt.Convolve(L) as conv:
        
        # Variation 1: MICRO-TOPOGRAPHY (s=0.1)
        # Highlights tiny surface bumps, noise, and sharpest edges.
        print("Processing: Micro-Topography...")
        sig_micro = get_saliency_signal(conv, coords, scale=0.1, percentile=85, power=1.2)
        
        # Variation 2: ANATOMICAL RIDGES (s=1.0)
        # Highlights the 'musculature', eyes, and defined creases.
        print("Processing: Anatomical Ridges...")
        sig_ridges = get_saliency_signal(conv, coords, scale=1.0, percentile=75, power=1.5)
        
        # Variation 3: STRUCTURAL EXTREMITIES (s=5.0)
        # Highlights larger features like the ears, tail, and paws.
        print("Processing: Structural Extremities...")
        sig_structure = get_saliency_signal(conv, coords, scale=5.0, percentile=65, power=2.0)
        
        # Variation 4: MULTI-SCALE SALIENCY (Combined)
        # A weighted 'heat map' of all geometric interestingness.
        print("Processing: Multi-scale Saliency...")
        sig_combo = (sig_micro * 0.3) + (sig_ridges * 0.5) + (sig_structure * 0.2)
        sig_combo = sig_combo / sig_combo.max()

    # --- 3. VISUALIZATION ---
    
    # Settings for the 'Activated' look:
    # We use Diverging colormaps (0 is centered/white)
    # or Sequential colormaps that start at white.
    configs = [
        (sig_micro,     "Red Activated Micro-Texture", "Reds",    "bunny_v1_micro.png"),
        (sig_ridges,    "Blue Activated Anatomy",      "Blues",   "bunny_v2_ridges.png"),
        (sig_structure, "Purple Activated Structure", "Purples", "bunny_v3_structure.png"),
        (sig_combo,     "Spectral Geometric Saliency", "magma",   "bunny_v4_saliency.png")
    ]

    for signal, title, cmap, filename in configs:
        print(f"Plotting {title}...")
        
        # Use the provided plot_mesh_wavelet_custom_coords function
        # Note: 'bwr' or 'seismic' are excellent for white-centered signals
        # but 'Reds'/'Blues' also work if we ensure the 0-value is pure white.
        plot_mesh_wavelet_custom_coords(
            signal=signal,
            coordinates=sgwt.BUNNY_XYZ,
            mesh="BUNNY",
            title=title,
            output_filename=Path(filename),
            cmap=cmap,
            mesh_rotation=(0, -90, 0), # Stand the bunny up
            elev=25,
            azims=[-45, 45, 135],
            zoom=1.5
        )

if __name__ == "__main__":
    generate_bunny_variations()