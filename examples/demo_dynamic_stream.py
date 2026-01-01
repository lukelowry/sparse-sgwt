import os
import matplotlib.pyplot as plt
import numpy as np

# Set font to Times New Roman for a professional look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12 # Increase base font size
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 16 # For suptitle if used


# --- Helper Function for Stream Simulation ---

# Sparse topology events throughout the stream, known a priori
events_data = {
    150: (1000, 5000, 1.0),
    350: (2000, 6000, 1.0),
    400: (3000, 7000, 1.0),
    420: (3000, 7001, 1.0),
    450: (3000, 7002, 1.0),
    500: (3000, 7003, 1.0),
    550: (3000, 7004, 1.0),
    750: (4000, 8000, 1.0),
    950: (5000, 9000, 1.0)
}
first_event_time = next(iter(events_data.keys())) if events_data else None
def get_incoming_data(t):
    """Mock signal generator and network event simulator."""
    n_nodes = L.shape[0]
    # Generate random signal (Fortran order for CHOLMOD efficiency)
    f_t = np.asfortranarray(np.random.randn(n_nodes, 1).astype(np.float64))
    event = events_data.get(t)
    
    return f_t, event

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import DyConvolve
from sgwt import DELAY_USA as L

# --- Configuration ---
scales = np.geomspace(0.1, 10.0, 10)
poles  = 1.0 / scales
N_SAMPLES = 1000

print("SGWT Online Processor Emulation")
print(f"Graph:  Synthetic USA ({L.shape[0]} nodes)")
print(f"Stream: {N_SAMPLES} samples\n")

avg_signal_magnitudes = []

with DyConvolve(L, poles) as conv:
    for t in range(N_SAMPLES):
        f_t, event = get_incoming_data(t)

        if event:
            u, v, w = event
            conv.addbranch(*event)
            print(f"[{t:04d}] EVENT | Topology Update: Edge ({u} <-> {v}) added")

        # Compute wavelet coefficients
        W = conv.bandpass(f_t)
        avg_signal_magnitudes.append(np.mean(np.abs(f_t))) # Record average signal magnitude
        
        if not event: print(f"[{t:04d}] STATUS | Stream processing active")

# DOC_END_CODE_EXCLUDE_PLOT
# Create a plot to visualize stream processing
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(range(N_SAMPLES), avg_signal_magnitudes, label='Average Signal Magnitude', color='blue', alpha=0.7)
for et in events_data.keys():
    ax.axvline(et, color='red', linestyle='--', alpha=0.6, label='Topology Event' if et == first_event_time else "")

ax.set_xlabel('Time Step')
ax.set_ylabel('Average Signal Magnitude')
ax.set_title('Dynamic GSP: Online Signal and Topology Events')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xlim(0, N_SAMPLES)

plt.tight_layout()

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_dynamic_stream.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()