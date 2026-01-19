"""
Mesh Wavelet Visualization - Stanford Bunny
============================================

This example demonstrates how to apply a spectral graph wavelet to the
Stanford Bunny mesh and visualize the result.
"""
from pathlib import Path

# DOC_START_CODE_EXCLUDE_IMPORTS
import sgwt


# --- Stanford Bunny Example ---
L_bunny = sgwt.MESH_BUNNY
bunny_impulse_node = 15000
bunny_scale = 200

x_bunny = sgwt.impulse(L_bunny, n=bunny_impulse_node)
with sgwt.Convolve(L_bunny) as conv:
    y_bunny = conv.bandpass(x_bunny, [bunny_scale], order=4)[0]

# DOC_END_CODE_EXCLUDE_PLOT

print("GSP Done! Begin Rendering")

# The plotting code is in a separate file and not rendered in the documentation.
from demo_mesh_plot import plot_mesh_wavelet

# Define output directory relative to this script's location
output_dir = Path(__file__).parent.parent / "docs/_static/images"
output_dir.mkdir(parents=True, exist_ok=True)

# Plot and save Bunny
bny_path = r"C:\Users\wyattluke.lowery\Documents\GitHub\laplib\StanfardBunny\src\Stanford Bunny\reconstruction\bun_zipper.ply"
plot_mesh_wavelet(
    y_bunny, bny_path ,"",
    output_dir / "demo_mesh_wavelet_1.png",
)
