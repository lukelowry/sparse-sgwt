module SpectralGraphWavelet

using MAT
using SparseArrays
using LinearAlgebra
using SuiteSparse
using Statistics
using Plots

# Include source files
include("GraphIO.jl")
include("GraphFilters.jl")
include("GraphPlot.jl")

# Export public API
export load_laplacian, load_signal
export plot_signal
export VFKern, DyConvolve, impulse, convolve, lowpass, bandpass, highpass, addbranch!

end