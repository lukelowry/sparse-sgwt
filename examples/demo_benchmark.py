# -*- coding: utf-8 -*-
"""
Example: Analytical Solver Performance Benchmarks

This demo visualizes performance benchmarks for the analytical graph wavelet
solvers, comparing Static (`Convolve`) and Dynamic (`DyConvolve`) approaches
across various graph sizes and parameter configurations.

Reads '.benchmarks/benchmark_results.json' from the project root.

If no benchmark data exists, the script will prompt you to run the benchmarks.
Alternatively, you can run them manually:
    pytest sgwt/tests/test_performance.py -m benchmark \\
        --benchmark-json=.benchmarks/benchmark_results.json
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# Professional Plotting Style (consistent with other examples)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
})

# DOC_START_CODE_EXCLUDE_IMPORTS
# Get project root (parent of examples folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RESULTS_FILE = os.path.join(PROJECT_ROOT, ".benchmarks", "benchmark_results.json")

# Graph sizes for known fixtures
GRAPH_SIZES = {
    "medium_path_laplacian": 100,
    "large_path_laplacian": 1000,
    "texas_laplacian": 2000,
    "hawaii_laplacian": 37,
    "wecc_laplacian": 240,
    "eastwest_laplacian": 65000,
    "usa_laplacian": 82223,
}

# Number of edges for known fixtures (undirected graphs, each edge counted once)
GRAPH_EDGES = {
    "medium_path_laplacian": 99,        # Path graph: N-1 edges
    "large_path_laplacian": 999,        # Path graph: N-1 edges
    "texas_laplacian": 3667,            # DELAY_TEXAS.nnz // 2
    "hawaii_laplacian": 90,             # DELAY_HAWAII.nnz // 2
    "wecc_laplacian": 472,              # DELAY_WECC.nnz // 2
    "eastwest_laplacian": 135544,       # DELAY_EASTWEST.nnz // 2
    "usa_laplacian": 139205,            # DELAY_USA.nnz // 2
}

GRAPH_LABELS = {
    37: "Hawaii",
    100: "Path-100",
    240: "WECC",
    1000: "Path-1k",
    2000: "Texas",
    65000: "EastWest",
    82223: "USA",
}

# Color scheme
COLORS = {
    'lowpass': '#2563eb',    # Blue
    'bandpass': '#059669',   # Green
    'highpass': '#d97706',   # Amber
}

MARKERS = {'lowpass': 'o', 'bandpass': 's', 'highpass': '^'}


def load_benchmark_data():
    """Load benchmark data from JSON file."""
    if not os.path.exists(RESULTS_FILE):
        return None
    try:
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read benchmark file: {e}")
        return None


def get_graph_size(graph_name):
    """Map graph fixture name to node count."""
    return GRAPH_SIZES.get(graph_name)


def get_graph_edges(graph_name):
    """Map graph fixture name to edge count."""
    return GRAPH_EDGES.get(graph_name)


def setup_style():
    """Configure matplotlib for clean, professional plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
    })


def extract_graph_scaling_data(benchmarks):
    """Extract graph edge scaling data for Static and Dynamic solvers."""
    static_data = {}
    dynamic_data = {}

    for b in benchmarks:
        name = b['name']
        params = b.get('params')
        if params is None:
            continue

        graph_name = params.get('graph_name')
        # Use number of edges from lookup table
        size = get_graph_edges(graph_name)
        if size is None:
            continue

        mean_time = b['stats']['mean']

        if 'test_static_convolution[' in name:
            method = params.get('method_name', 'unknown')
            if method not in static_data:
                static_data[method] = {}
            if size not in static_data[method]:
                static_data[method][size] = []
            static_data[method][size].append(mean_time)

        elif 'test_dynamic_convolution[' in name:
            method = params.get('method_name', 'unknown')
            if method not in dynamic_data:
                dynamic_data[method] = {}
            if size not in dynamic_data[method]:
                dynamic_data[method][size] = []
            dynamic_data[method][size].append(mean_time)

    return static_data, dynamic_data


def extract_parameter_scaling(benchmarks, test_prefix, param_name):
    """Extract parameter scaling data for a specific test pattern."""
    groups = {}

    for b in benchmarks:
        name = b['name']
        if test_prefix not in name:
            continue

        params = b.get('params')
        if params is None:
            continue

        param_val = params.get(param_name)
        if param_val is None:
            continue

        if 'test_order_scaling_bandpass' in name:
            method = 'Bandpass'
        elif 'method_name' in params:
            method = params['method_name'].title()
        else:
            method = 'Unknown'

        if '_dynamic' in name:
            label = f"Dynamic {method}"
        else:
            label = f"Static {method}"

        if label not in groups:
            groups[label] = {'x': [], 'y': [], 'err': []}

        groups[label]['x'].append(float(param_val))
        groups[label]['y'].append(b['stats']['mean'])
        groups[label]['err'].append(b['stats']['stddev'])

    return groups


def plot_graph_scaling(ax, static_data, dynamic_data, y_lim=None):
    """Plot graph size scaling: one line per filter, comparing Static vs Dynamic."""
    filter_types = ['lowpass', 'bandpass', 'highpass']

    for method in filter_types:
        color = COLORS.get(method, '#666666')
        marker = MARKERS.get(method, 'o')

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

    if y_lim:
        ax.set_ylim(y_lim)


def plot_signal_scaling(ax, signal_data, y_lim=None):
    """Plot signal count scaling on a log-log scale."""
    filter_types = ['Lowpass', 'Bandpass', 'Highpass']
    solvers = ['Static', 'Dynamic']

    for method in filter_types:
        for solver in solvers:
            label = f"{solver} {method}"
            if label not in signal_data or len(signal_data[label]['x']) < 2:
                continue

            d = signal_data[label]
            sorted_idx = np.argsort(d['x'])
            x = np.array(d['x'])[sorted_idx]
            y = np.array(d['y'])[sorted_idx]

            color = COLORS.get(method.lower(), '#666666')
            marker = MARKERS.get(method.lower(), 'o')
            linestyle = '--' if solver == 'Dynamic' else '-'
            mfc = 'white' if solver == 'Dynamic' else color

            ax.loglog(x, y, marker=marker, linestyle=linestyle, color=color,
                      label=label, markersize=6, linewidth=1.8,
                      markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=1.2)

    ax.set_xlabel('Number of Signals (M)')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Complexity vs. Signal Count (M)')
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    if y_lim:
        ax.set_ylim(y_lim)


def plot_scale_scaling(ax, scale_data, y_lim=None):
    """Plot scale count scaling on a log-log scale."""
    filter_types = ['Lowpass', 'Bandpass', 'Highpass']
    solvers = ['Static', 'Dynamic']

    for method in filter_types:
        for solver in solvers:
            label = f"{solver} {method}"
            if label not in scale_data or len(scale_data[label]['x']) < 2:
                continue

            d = scale_data[label]
            sorted_idx = np.argsort(d['x'])
            x = np.array(d['x'])[sorted_idx]
            y = np.array(d['y'])[sorted_idx]

            color = COLORS.get(method.lower(), '#666666')
            marker = MARKERS.get(method.lower(), 'o')
            linestyle = '--' if solver == 'Dynamic' else '-'
            mfc = 'white' if solver == 'Dynamic' else color

            ax.loglog(x, y, marker=marker, linestyle=linestyle, color=color,
                      label=label, markersize=6, linewidth=1.8,
                      markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=1.2)

    ax.set_xlabel('Number of Scales (J)')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Complexity vs. Scale Count (J)')
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    if y_lim:
        ax.set_ylim(y_lim)


def plot_bandpass_order_scaling(ax, order_data, y_lim=None):
    """Plot bandpass filter order scaling on a log-log scale."""
    if not order_data:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    solvers = ['Static', 'Dynamic']

    for solver in solvers:
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

    if y_lim:
        ax.set_ylim(y_lim)


def run_benchmarks():
    """Run the performance benchmarks and return the data."""
    import subprocess
    import sys

    print("Running benchmarks (this may take several minutes)...")
    benchmarks_dir = os.path.join(PROJECT_ROOT, ".benchmarks")
    os.makedirs(benchmarks_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest",
        os.path.join(PROJECT_ROOT, "sgwt", "tests", "test_performance.py"),
        "-m", "benchmark",
        f"--benchmark-json={RESULTS_FILE}"
    ]

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print("\nBenchmarks completed. Loading results...\n")
        return load_benchmark_data()
    except subprocess.CalledProcessError as e:
        print(f"\nError running benchmarks: {e}")
        return None


data = load_benchmark_data()
if data is None:
    print("No benchmark data found. Running benchmarks automatically...")
    data = run_benchmarks()

if data is None:
    print("Error: Could not load or generate benchmark data.")
else:
    benchmarks = data['benchmarks']
    setup_style()

    # Extract data
    static_data, dynamic_data = extract_graph_scaling_data(benchmarks)
    signal_data = extract_parameter_scaling(benchmarks, "test_signal_scaling", "n_signals")
    scale_data = extract_parameter_scaling(benchmarks, "test_scale_scaling", "n_scales")
    order_data = extract_parameter_scaling(benchmarks, "test_order_scaling_bandpass", "order")

    # Find common y-axis limits for top row
    top_row_times = []
    if 'lowpass' in static_data:
        top_row_times.extend([np.mean(static_data['lowpass'][s]) for s in static_data['lowpass']])
    if 'lowpass' in dynamic_data:
        top_row_times.extend([np.mean(dynamic_data['lowpass'][s]) for s in dynamic_data['lowpass']])
    for group in signal_data.values():
        top_row_times.extend(group['y'])

    top_y_lim = None
    if top_row_times:
        min_y, max_y = min(top_row_times), max(top_row_times)
        log_min, log_max = np.log10(min_y), np.log10(max_y)
        padding = (log_max - log_min) * 0.1
        top_y_lim = (10**(log_min - padding), 10**(log_max + padding))

    # Find common y-axis limits for bottom row
    bottom_row_times = []
    for group in scale_data.values():
        bottom_row_times.extend(group['y'])
    for group in order_data.values():
        bottom_row_times.extend(group['y'])

    bottom_y_lim = None
    if bottom_row_times:
        min_y, max_y = min(bottom_row_times), max(bottom_row_times)
        if np.isclose(min_y, max_y):
            min_y *= 0.1
            max_y *= 10
        log_min, log_max = np.log10(min_y), np.log10(max_y)
        padding = (log_max - log_min) * 0.1
        bottom_y_lim = (10**(log_min - padding), 10**(log_max + padding))
# DOC_END_CODE_EXCLUDE_PLOT

    # Create Summary Dashboard (2x2)
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle("Analytical Solver Performance Benchmarks", fontsize=16, fontweight='bold')

    plot_graph_scaling(ax_a, static_data, dynamic_data, y_lim=top_y_lim)
    plot_signal_scaling(ax_b, signal_data, y_lim=top_y_lim)
    plot_scale_scaling(ax_c, scale_data, y_lim=bottom_y_lim)
    plot_bandpass_order_scaling(ax_d, order_data, y_lim=bottom_y_lim)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure for documentation
    static_images_dir = os.path.join(PROJECT_ROOT, 'docs', '_static', 'images')
    os.makedirs(static_images_dir, exist_ok=True)
    save_path = os.path.join(static_images_dir, 'demo_benchmark.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()
