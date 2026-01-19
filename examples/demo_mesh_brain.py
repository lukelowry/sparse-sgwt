"""
Mesh Wavelet Visualization - Brain
==================================

This example demonstrates how to apply a spectral graph wavelet to a
brain mesh and visualize the result.
"""
from pathlib import Path

# DOC_START_CODE_EXCLUDE_IMPORTS
import sgwt

L_brain = sgwt.MESH_LBRAIN
brain_impulse_node = 70000
brain_scale = 200

x_brain = sgwt.impulse(L_brain, n=brain_impulse_node)
with sgwt.Convolve(L_brain) as conv:
    y_brain = conv.bandpass(x_brain, [brain_scale], order=40)[0]
# DOC_END_CODE_EXCLUDE_PLOT

print("GSP Done! Begin Rendering")


# The plotting code is in a separate file and not rendered in the documentation.
from demo_mesh_plot import plot_mesh_wavelet

# Define output directory relative to this script's location
output_dir = Path(__file__).parent.parent / "docs/_static/images"
output_dir.mkdir(parents=True, exist_ok=True)

# Plot and save Brain (uses bundled PLY file)
plot_mesh_wavelet(
    y_brain, "LBRAIN", "",
    output_dir / "demo_mesh_wavelet_3.png",
    mesh_rotation=(-90, 0, 0),
    elev=15,
    azims=[0, 180, 120],
)
