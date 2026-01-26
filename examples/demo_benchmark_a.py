# -*- coding: utf-8 -*-
"""
Example: Graph Size Scaling Benchmark

This demo visualizes how execution time scales with graph size (number of edges)
for the analytical graph wavelet solvers, comparing Static (Convolve) and
Dynamic (DyConvolve) approaches across lowpass, bandpass, and highpass filters.

Reads from '.benchmarks/benchmark_graph.json' (or combined file as fallback).
"""
import numpy as np
import matplotlib.pyplot as plt

from benchmark_utils import (
    COLORS, MARKERS, FILTER_TYPES,
    setup_style, save_figure, ensure_data_loaded, extract_graph_scaling,
    print_graph_benchmark_summary
)

# DOC_START_CODE_EXCLUDE_IMPORTS


def plot_graph_scaling(ax, static_data, dynamic_data):
    """Plot graph size scaling: one line per filter, comparing Static vs Dynamic."""
    for method in FILTER_TYPES:
        color = COLORS[method]
        marker = MARKERS[method]

        if method in static_data and static_data[method]:
            sizes = sorted(static_data[method].keys())
            times = [np.mean(static_data[method][s]) for s in sizes]
            ax.loglog(sizes, times, marker=marker, linestyle='-', color=color,
                      label=f'Static {method.title()}', markersize=6, linewidth=1.8,
                      markerfacecolor=color, markeredgecolor=color)

        if method in dynamic_data and dynamic_data[method]:
            sizes = sorted(dynamic_data[method].keys())
            times = [np.mean(dynamic_data[method][s]) for s in sizes]
            ax.loglog(sizes, times, marker=marker, linestyle='--', color=color,
                      label=f'Dynamic {method.title()}', markersize=6, linewidth=1.8,
                      alpha=0.85, markerfacecolor='white', markeredgecolor=color,
                      markeredgewidth=1.2)

    ax.set_xlabel('Number of Edges')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Static vs. Dynamic Solver Scaling')
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, linestyle='--')


# DOC_END_CODE_EXCLUDE_PLOT

if __name__ == '__main__':
    benchmarks = ensure_data_loaded('graph')
    setup_style()

    static_data, dynamic_data = extract_graph_scaling(benchmarks)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_graph_scaling(ax, static_data, dynamic_data)
    plt.tight_layout()

    print_graph_benchmark_summary(static_data, dynamic_data)

    save_figure(fig, 'demo_benchmark_a.png')
    plt.show()
