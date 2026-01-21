# -*- coding: utf-8 -*-
"""
Example: Signal Count Scaling Benchmark

This demo visualizes how execution time scales with the number of input signals (M)
for the analytical graph wavelet solvers.

Reads from '.benchmarks/benchmark_signal.json' (or combined file as fallback).
"""
import matplotlib.pyplot as plt

from benchmark_utils import (
    setup_style, save_figure, ensure_data_loaded,
    extract_scaling_data, plot_scaling_comparison
)

# DOC_START_CODE_EXCLUDE_IMPORTS


def plot_signal_scaling(ax, data):
    """Plot signal count scaling on a log-log scale."""
    plot_scaling_comparison(
        ax, data,
        xlabel='Number of Signals (M)',
        ylabel='Execution Time (s)',
        title='Complexity vs. Signal Count (M)'
    )


# DOC_END_CODE_EXCLUDE_PLOT

if __name__ == '__main__':
    benchmarks = ensure_data_loaded('signal')
    setup_style()

    signal_data = extract_scaling_data(benchmarks, 'n_signals')

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_signal_scaling(ax, signal_data)
    plt.tight_layout()

    save_figure(fig, 'demo_benchmark_b.png')
    plt.show()
