"""
Mesh Wavelet Visualization - Horse
==================================

This example demonstrates how to apply a spectral graph wavelet to a
horse mesh and visualize the result.
"""
from pathlib import Path
import numpy as np

# DOC_START_CODE_EXCLUDE_IMPORTS
import sgwt

L_horse = sgwt.MESH_HORSE
horse_impulse_node = 28000
horse_scale = 60

x_horse = sgwt.impulse(L_horse, n=horse_impulse_node)
with sgwt.Convolve(L_horse) as conv:
    y_horse = conv.bandpass(x_horse, [horse_scale], order=50)[0]
# DOC_END_CODE_EXCLUDE_PLOT

print("GSP Done! Begin Rendering")


# The plotting code is in a separate file and not rendered in the documentation.
from demo_mesh_plot import plot_mesh_wavelet

# Define output directory relative to this script's location
output_dir = Path(__file__).parent.parent / "docs/_static/images"
output_dir.mkdir(parents=True, exist_ok=True)

# Plot and save Horse (uses bundled PLY file)
plot_mesh_wavelet(
    y_horse, "HORSE", "",
    output_dir / "demo_mesh_wavelet_2.png",
    mesh_rotation=(-90, -90, 0),
    elev=15,
    azims=[-180, 0, 100],
    light_dir=np.array([0.3, 0.3, 1.0]),  
    zoom=1.3
)