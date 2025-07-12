from ..src.kernel.main import KernelDesign
from scipy.sparse import csc_matrix, save_npz

# Example to demonstrate how to design the SGWT kernel

# Graph Laplacian
L = None # wb.length_laplacian()*(1/2)

# Save 
save_npz('LAP.npz', csc_matrix(L))

# Design the Kernel
kern = KernelDesign(
    spectrum_range = (1e-7, 1e1),
    scale_range    = (1e2, 5e5), # (5e3, 1e5),
    pole_min       = 1e-6,
    nscales = 65,
    npoles  = 25,
    nsamples = 400,
    order = 1
)

# Save the Kernel
kern.write('kernel_model')