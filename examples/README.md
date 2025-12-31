# SGWT Examples

This directory contains demonstration scripts for various features of the `sgwt` package.

## Basic Usage
- `demo_filters_1.py`: Basic low-pass, band-pass, and high-pass filtering on the Texas grid.
- `demo_filters_2.py`: Filtering on the large East-West US grid (~65k nodes).
- `demo_filters_3.py`: Filtering on the full USA grid.
- `demo_single_file.py`: Minimal end-to-end example of band-pass filtering.

## Advanced Convolution
- `demo_vf.py`: Using Vector Fitting (VF) kernels for custom filter shapes (e.g., Modified Morlet).
- `demo_recon.py`: Signal reconstruction (e.g., coordinate recovery) from sparse measurements.
- `demo_inpainting.py`: Reconstructing a smooth signal from a small fraction of samples using iterative low-pass filtering.

## Dynamic Graphs
- `demo_dynamic_topology.py`: Updating graph topology (adding branches) on-the-fly with `DyConvolve`.
- `demo_dynamic_time.py`: Performance comparison between static and dynamic convolution methods.

## Utilities
- `demo_plot.py`: Helper functions for visualizing graph signals using Matplotlib.
---
*Note: Advanced features like Time-Vertex Convolution and Local JWT are currently under development.*
