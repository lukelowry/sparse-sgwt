"""
Mesh Wavelet Visualization
==========================

This example demonstrates how to apply a spectral graph wavelet to a 3D mesh
(the Stanford Bunny and a horse model) and visualize the result.
"""
from pathlib import Path

# DOC_START_CODE_EXCLUDE_IMPORTS
import sgwt


# --- Stanford Bunny Example ---
L_bunny = sgwt.MESH_BUNNY
C_bunny = sgwt.BUNNY_XYZ
bunny_impulse_node = 15000
bunny_scale = 200

x_bunny = sgwt.impulse(L_bunny, n=bunny_impulse_node)
with sgwt.Convolve(L_bunny) as conv:
    y_bunny = conv.bandpass(x_bunny, [bunny_scale], order=4)[0]


# --- Horse Example ---
L_horse = sgwt.MESH_HORSE
C_horse = sgwt.HORSE_XYZ
horse_impulse_node = 40000
horse_scale = 90

x_horse = sgwt.impulse(L_horse, n=horse_impulse_node)
with sgwt.Convolve(L_horse) as conv:
    y_horse = conv.bandpass(x_horse, [horse_scale], order=50)[0]
# DOC_END_CODE_EXCLUDE_PLOT

# The plotting code is in a separate file and not rendered in the documentation.
from demo_mesh_plot import plot_mesh_wavelet

# Define output directory relative to this script's location
output_dir = Path(__file__).parent.parent / "docs/_static/images"
output_dir.mkdir(parents=True, exist_ok=True)

# Plot and save Bunny
bny_path = r"C:\Users\wyattluke.lowery\Documents\GitHub\laplib\StanfardBunny\src\Stanford Bunny\reconstruction\bun_zipper.ply"
#plot_mesh_wavelet(y_bunny, bny_path ,"", output_dir / "demo_mesh_wavelet_1.png")

# Plot and save Horse
horse_pth = r"C:\Users\wyattluke.lowery\Documents\GitHub\laplib\Horse\horse.ply"
plot_mesh_wavelet(y_horse, horse_pth,"", output_dir / "demo_mesh_wavelet_2.png")
