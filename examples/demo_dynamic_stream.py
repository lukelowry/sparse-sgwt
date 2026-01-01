import numpy as np
from sgwt.dynamic import DyConvolve
from sgwt import DELAY_USA as L

def get_incoming_data(t):
    """Mock signal generator and network event simulator."""
    n_nodes = L.shape[0]
    # Generate random signal (Fortran order for CHOLMOD efficiency)
    f_t = np.asfortranarray(np.random.randn(n_nodes, 1).astype(np.float64))

    # Sparse topology events throughout the stream
    events = {
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
    event = events.get(t)
    
    return f_t, event

# 1. Configuration
scales = np.geomspace(0.1, 10.0, 10)
poles  = 1.0 / scales
N_SAMPLES = 1000

print("SGWT Online Processor Emulation")
print(f"Graph:  Synthetic USA ({L.shape[0]} nodes)")
print(f"Stream: {N_SAMPLES} samples\n")

# 2. Execution Context
with DyConvolve(L, poles) as conv:
    for t in range(N_SAMPLES):
        f_t, event = get_incoming_data(t)

        if event:
            u, v, w = event
            conv.addbranch(*event)
            print(f"[{t:04d}] \033[93mEVENT\033[0m  | Topology Update: Edge ({u} <-> {v}) added")

        # Compute wavelet coefficients
        W = conv.bandpass(f_t)
        
        if not event:
            print(f"[{t:04d}] STATUS | Stream processing active")

print("\nStream processing complete.")