using Test
using SpectralGraphWavelet
using SparseArrays

@testset "SpectralGraphWavelet Tests" begin
    # Create a simple graph (Line graph of 4 nodes)
    # 1 - 2 - 3 - 4
    # Laplacian:
    #  1 -1  0  0
    # -1  2 -1  0
    #  0 -1  2 -1
    #  0  0 -1  1
    
    I = [1, 2, 2, 3, 3, 4, 1, 2, 3, 4]
    J = [2, 1, 3, 2, 4, 3, 1, 2, 3, 4]
    V = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 1.0]
    L = sparse(I, J, V, 4, 4)
    
    @testset "GraphFilters" begin
        # Mock Kernel (Simple pole/residue)
        Q = [-1.0]
        R = [1.0]
        K = VFKern(Q, R)
        
        conv = DyConvolve(L, K)
        @test conv isa DyConvolve
        
        # Test impulse
        b = impulse(L, 1, 1)
        @test size(b) == (4, 1)
        @test b[1, 1] == 1.0
        
        # Test convolve
        res = convolve(conv, b, K)
        @test size(res) == (4, 1)
        
        # Test filters (smoke tests)
        @test length(lowpass(conv, b)) == 1
        @test length(bandpass(conv, b)) == 1
        @test length(highpass(conv, b)) == 1
    end
end