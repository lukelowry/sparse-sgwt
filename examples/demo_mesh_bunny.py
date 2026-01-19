"""
Mesh Wavelet Visualization - Stanford Bunny
============================================

This example demonstrates how to apply a spectral graph wavelet to the
Stanford Bunny mesh and visualize the result.
"""
from pathlib import Path

import numpy as np

# DOC_START_CODE_EXCLUDE_IMPORTS
import sgwt

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

# Plot and save Bunny (uses bundled PLY file)
plot_mesh_wavelet(
    y_bunny, "BUNNY", "",
    output_dir / "demo_mesh_wavelet_1.png",
    mesh_rotation=(0, -90, 0),
    zoom=1.3,
    light_dir=np.array([0.3, 0.3, 1.0])
)
