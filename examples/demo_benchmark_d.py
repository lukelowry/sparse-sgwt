# -*- coding: utf-8 -*-
"""
Example: Filter Order Scaling Benchmark

This demo visualizes how bandpass filter order (K) affects execution time
for the analytical graph wavelet solvers.

Reads from '.benchmarks/benchmark_order.json' (or combined file as fallback).
"""
import numpy as np
import matplotlib.pyplot as plt

from benchmark_utils import (
    COLORS, MARKERS, SOLVERS,
    setup_style, save_figure, ensure_data_loaded, extract_scaling_data
)

# DOC_START_CODE_EXCLUDE_IMPORTS


def plot_bandpass_order_scaling(ax, order_data):
    """Plot bandpass filter order scaling on a log-log scale."""
    if not order_data:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    for solver in SOLVERS:
        label = f"{solver} Bandpass"
        if label not in order_data or not order_data[label]['x']:
            continue

        d = order_data[label]
        sorted_idx = np.argsort(d['x'])
        x = np.array(d['x'])[sorted_idx]
        y = np.array(d['y'])[sorted_idx]

        linestyle = '--' if solver == 'Dynamic' else '-'
        mfc = 'white' if solver == 'Dynamic' else COLORS['bandpass']

        ax.loglog(x, y, marker=MARKERS['bandpass'], linestyle=linestyle,
                  color=COLORS['bandpass'], markersize=7, linewidth=2, label=label,
                  markerfacecolor=mfc, markeredgecolor=COLORS['bandpass'],
                  markeredgewidth=1.2)

    ax.set_xlabel('Filter Order (K)')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Complexity vs. Bandpass Order (K)')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')


# DOC_END_CODE_EXCLUDE_PLOT

if __name__ == '__main__':
    benchmarks = ensure_data_loaded('order')
    setup_style()

    order_data = extract_scaling_data(benchmarks, 'order')

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_bandpass_order_scaling(ax, order_data)
    plt.tight_layout()

    save_figure(fig, 'demo_benchmark_d.png')
    plt.show()
