# -*- coding: utf-8 -*-
"""
Example: Scale Count Scaling Benchmark

This demo visualizes how execution time scales with the number of wavelet scales (J)
for the analytical graph wavelet solvers.

Reads from '.benchmarks/benchmark_scale.json' (or combined file as fallback).
"""
import matplotlib.pyplot as plt

from benchmark_utils import (
    setup_style, save_figure, ensure_data_loaded,
    extract_scaling_data, plot_scaling_comparison,
    print_benchmark_summary
)

# DOC_START_CODE_EXCLUDE_IMPORTS


def plot_scale_scaling(ax, data):
    """Plot scale count scaling on a log-log scale."""
    plot_scaling_comparison(
        ax, data,
        xlabel='Number of Scales (J)',
        ylabel='Execution Time (s)',
        title='Complexity vs. Scale Count (J)'
    )


# DOC_END_CODE_EXCLUDE_PLOT

if __name__ == '__main__':
    benchmarks = ensure_data_loaded('scale')
    setup_style()

    scale_data = extract_scaling_data(benchmarks, 'n_scales')

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_scale_scaling(ax, scale_data)
    plt.tight_layout()

    print_benchmark_summary(scale_data, param_name="Scales (J)")

    save_figure(fig, 'demo_benchmark_c.png')
    plt.show()
