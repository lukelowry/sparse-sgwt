# -*- coding: utf-8 -*-
"""
Shared Utilities for Benchmark Examples

This module provides common constants, data loading, and styling functions
used across the individual benchmark plotting scripts.
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

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BENCHMARKS_DIR = os.path.join(PROJECT_ROOT, ".benchmarks")
STATIC_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'docs', '_static', 'images')

# Individual result files for each benchmark category
RESULTS_FILES = {
    'graph': os.path.join(BENCHMARKS_DIR, "benchmark_graph.json"),
    'signal': os.path.join(BENCHMARKS_DIR, "benchmark_signal.json"),
    'scale': os.path.join(BENCHMARKS_DIR, "benchmark_scale.json"),
    'order': os.path.join(BENCHMARKS_DIR, "benchmark_order.json"),
}

# Pytest filter patterns for each category (used with -k option)
TEST_PATTERNS = {
    'graph': 'test_static_convolution or test_dynamic_convolution',
    'signal': 'test_signal_scaling',
    'scale': 'test_scale_scaling',
    'order': 'test_order_scaling_bandpass',
}

# ---------------------------------------------------------------------------
# Graph Constants - Dynamic Loading
# ---------------------------------------------------------------------------
def get_graph_stats_from_laplacian(graph_name):
    """
    Dynamically retrieve node and edge count from a Laplacian fixture.

    Parameters
    ----------
    graph_name : str
        Fixture name (e.g., 'texas_laplacian')

    Returns
    -------
    dict or None
        Dictionary with keys: 'nodes', 'edges', 'label', or None if not found
    """
    import sgwt

    # Map fixture names to sgwt constants
    FIXTURE_TO_CONSTANT = {
        'texas_laplacian': sgwt.DELAY_TEXAS,
        'hawaii_laplacian': sgwt.DELAY_HAWAII,
        'wecc_laplacian': sgwt.DELAY_WECC,
        'eastwest_laplacian': sgwt.DELAY_EASTWEST,
        'usa_laplacian': sgwt.DELAY_USA,
    }

    if graph_name not in FIXTURE_TO_CONSTANT:
        return None

    L = FIXTURE_TO_CONSTANT[graph_name]
    n_nodes = L.shape[0]
    n_edges = (L.nnz - np.count_nonzero(L.diagonal())) // 2

    # Generate label from fixture name
    label = graph_name.replace('_laplacian', '').replace('_', ' ').title()

    return {'nodes': n_nodes, 'edges': n_edges, 'label': label}

# ---------------------------------------------------------------------------
# Plot Style Constants
# ---------------------------------------------------------------------------
COLORS = {
    'lowpass': '#2563eb',
    'bandpass': '#059669',
    'highpass': '#d97706',
}

MARKERS = {'lowpass': 'o', 'bandpass': 's', 'highpass': '^'}

FILTER_TYPES = ['lowpass', 'bandpass', 'highpass']
SOLVERS = ['Static', 'Dynamic']


# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------
def _load_json(filepath):
    """Load JSON file, returning None on failure."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None


def _save_json(filepath, data):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_category(category):
    """
    Load benchmark data for a specific category.

    Parameters
    ----------
    category : str
        One of: 'graph', 'signal', 'scale', 'order'

    Returns
    -------
    list or None
        List of benchmark entries for the category, or None if not found.
    """
    if category not in RESULTS_FILES:
        raise ValueError(f"Unknown category: {category}")

    return _load_json(RESULTS_FILES[category])


# ---------------------------------------------------------------------------
# Graph Lookup Functions
# ---------------------------------------------------------------------------
def get_graph_size(graph_name):
    """Map graph fixture name to node count (dynamic)."""
    stats = get_graph_stats_from_laplacian(graph_name)
    return stats['nodes'] if stats else None


def get_graph_edges(graph_name):
    """Map graph fixture name to edge count (dynamic)."""
    stats = get_graph_stats_from_laplacian(graph_name)
    return stats['edges'] if stats else None


# ---------------------------------------------------------------------------
# Plot Setup
# ---------------------------------------------------------------------------
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


def save_figure(fig, filename):
    """Save figure to the static images directory."""
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
    filepath = os.path.join(STATIC_IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    return filepath


# ---------------------------------------------------------------------------
# Data Extraction Helpers
# ---------------------------------------------------------------------------
def extract_scaling_data(benchmarks, param_name):
    """
    Extract parameter scaling data grouped by solver and filter type.

    Parameters
    ----------
    benchmarks : list
        List of benchmark entries.
    param_name : str
        Parameter to extract (e.g., 'n_signals', 'n_scales', 'order').

    Returns
    -------
    dict
        Data grouped by "{Solver} {Filter}" labels.
    """
    groups = {}

    for b in benchmarks:
        name = b['name']
        params = b.get('params')
        if params is None:
            continue

        param_val = params.get(param_name)
        if param_val is None:
            continue

        # Determine filter type
        if 'test_order_scaling_bandpass' in name:
            method = 'Bandpass'
        elif 'method_name' in params:
            method = params['method_name'].title()
        else:
            continue

        # Determine solver type
        solver = 'Dynamic' if '_dynamic' in name else 'Static'
        label = f"{solver} {method}"

        if label not in groups:
            groups[label] = {'x': [], 'y': [], 'err': []}

        groups[label]['x'].append(float(param_val))
        groups[label]['y'].append(b['stats']['mean'])
        groups[label]['err'].append(b['stats']['stddev'])

    return groups


def extract_graph_scaling(benchmarks):
    """
    Extract graph scaling data grouped by solver type and filter.

    Returns
    -------
    static_data, dynamic_data : dict, dict
        Dictionaries mapping filter -> {edge_count: [times]}.
    """
    static_data = {}
    dynamic_data = {}

    for b in benchmarks:
        name = b['name']
        params = b.get('params')
        if params is None:
            continue

        graph_name = params.get('graph_name')
        edges = get_graph_edges(graph_name)
        if edges is None:
            continue

        method = params.get('method_name', 'unknown')
        mean_time = b['stats']['mean']

        if 'test_static_convolution[' in name:
            target = static_data
        elif 'test_dynamic_convolution[' in name:
            target = dynamic_data
        else:
            continue

        if method not in target:
            target[method] = {}
        if edges not in target[method]:
            target[method][edges] = []
        target[method][edges].append(mean_time)

    return static_data, dynamic_data


# ---------------------------------------------------------------------------
# Common Plotting Functions
# ---------------------------------------------------------------------------
def plot_scaling_comparison(ax, data, xlabel, ylabel, title):
    """
    Plot scaling data comparing Static vs Dynamic solvers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    data : dict
        Data from extract_scaling_data().
    xlabel, ylabel, title : str
        Axis labels and title.
    """
    for method in ['Lowpass', 'Bandpass', 'Highpass']:
        for solver in SOLVERS:
            label = f"{solver} {method}"
            if label not in data or len(data[label]['x']) < 2:
                continue

            d = data[label]
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------
def run_benchmarks(category):
    """
    Run benchmarks for a specific category.

    Parameters
    ----------
    category : str
        One of: 'graph', 'signal', 'scale', 'order'

    Returns
    -------
    list or None
        List of benchmark entries, or None on failure.
    """
    import subprocess
    import sys

    if category not in TEST_PATTERNS:
        raise ValueError(f"Unknown category: {category}")

    output_file = RESULTS_FILES[category]
    test_pattern = TEST_PATTERNS[category]

    print(f"Running {category} benchmarks...")
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest",
        os.path.join(PROJECT_ROOT, "sgwt", "tests", "test_performance.py"),
        "-m", "benchmark",
        "-k", test_pattern,
        f"--benchmark-json={output_file}"
    ]

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print(f"Benchmarks completed. Results saved to {os.path.basename(output_file)}\n")

        # Load and extract just the benchmarks list
        data = _load_json(output_file)
        if data and 'benchmarks' in data:
            benchmarks = data['benchmarks']
            _save_json(output_file, benchmarks)
            return benchmarks
        return data
    except subprocess.CalledProcessError as e:
        print(f"\nError running benchmarks: {e}")
        return None


def ensure_data_loaded(category):
    """
    Load benchmark data for a category, running benchmarks if necessary.

    Parameters
    ----------
    category : str
        One of: 'graph', 'signal', 'scale', 'order'

    Returns
    -------
    list
        List of benchmark entries.
    """
    if category not in RESULTS_FILES:
        raise ValueError(f"Unknown category: {category}")

    data = load_category(category)
    if data is not None:
        return data

    print(f"No {category} benchmark data found. Running benchmarks automatically...")
    data = run_benchmarks(category)

    if data is None:
        raise RuntimeError(f"Could not load or generate {category} benchmark data.")

    return data


# Backwards compatibility alias
extract_parameter_scaling = extract_scaling_data
