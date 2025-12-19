

from .analytic import AnalyticFilters

from ..cholesky import CholWrapper, cholmod_dense, cholmod_sparse

from scipy.sparse import csc_matrix
from ctypes import byref,  POINTER


class FiltersDLL(AnalyticFilters):
    '''
    A sparse memory efficient implementation
    that uses cholmod_solve2
    '''
    
    def __init__(self, L: csc_matrix, scales=[1]):
        '''
        Description: 
            A class that of analytical versions filters for SGWT and GSP
        Parameters:
            L: sparse csc_matrix form of Graph Laplacian (real valued)
            scales: optional, default scales used
        '''

        # Sparse Laplacian
        self.L = L

        # Discrete Scales
        self.scales = scales
        self.nscales = len(scales)

        # NOTE improve how this is used so I don't have to do this
        self.chol = CholWrapper(L)


    # Context Manager for using CHOLMOD
    def __enter__(self):

        # Start Cholmod
        self.chol.start()

        # Symbolically Factor
        self.chol.sym_factor()

        # Workspace for operations in solve2
        self.X1    = POINTER(cholmod_dense)()
        self.X2    = POINTER(cholmod_dense)()
        self.Xset  = POINTER(cholmod_sparse)()

        # Provide solve2 with re-usable workspace
        self.Y    = POINTER(cholmod_dense)()
        self.E    = POINTER(cholmod_dense)()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        # Free the factored matrix object
        self.chol.free_factor()

        # Free working memory used in solve2
        self.chol.free_dense(self.X1)
        self.chol.free_dense(self.X2)
        self.chol.free_sparse(self.Xset)

        # Free Y & E (workspacce for solve2)
        self.chol.free_dense(self.Y)
        self.chol.free_dense(self.E)

        # Finish cholmod
        self.chol.finish()

    '''
    Convolutions
    '''

    def scaling_coeffs2(self, f, fset=None, scales=None):
        '''
        Description
            Scaling coefficnets at indicated scales using the analytical form
            I/(aL+I)
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate scaling coeffs.
            fset: Used to solve for a sparse subset of coeffs. ncol must be 1
            scales: list (numScales) of scales to compute scaling coefficents for.
        Returns
            Scaling coefficients for each scale (numVerticies x numScales)
        '''
        # Scales
        scales = self.scales if scales is None else scales
  
        # Pointer to f (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(f))
        Bset = None

        # Using this requires the number of columns in f to be 1
        if fset is not None:
            Bset = byref(self.chol.numpy_to_chol_sparse_vec(fset))

        # My workspace pointers
        X1       = self.X1
        Xset     = self.Xset
        Y , E    = self.Y , self.E

        # Store Resulting matricies here
        W = []

        # For each wavelet scale
        for scale in self.scales:

            # Step 1 -> Set Scale via shifted numeric factor
            self.chol.num_factor(1/scale)

            # Step 2 -> Solve (L+beta*I) B = X1
            self.chol.solve2(B, Bset, X1, Xset, Y, E) 

            # Scale
            #self.chol.sdmult(X1, X2, 0, 1/scale)

            # Convert to numpy (copy)
            x_numpy = self.chol.chol_dense_to_numpy(X1)/scale
            
            # Append to list
            W.append(x_numpy)

        # Although returns full matrix, only rows specified by Bset are valid
        return W
    
    def _allocate_results(self):
        return []
    
    def _format_rhs(self, b, bset):

        # Pointer to b (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(b))

        # Using this requires the number of columns in f to be 1
        if bset is not None:
            Bset = byref(self.chol.numpy_to_chol_sparse_vec(bset))
        else:
            Bset = None
        
        return B, Bset

    
    def _save_to_results(self, x, index):

        self.results.append(
            self.chol.chol_dense_to_numpy(x)
        )
    
    def _numeric_factorization(self, beta):
        self.chol.num_factor(beta)

    def _solve(self, b, bset):
        self.chol.solve2(
            b, 
            bset, 
            self.X1, 
            self.Xset, 
            self.Y, 
            self.E
        ) 
        return self.X1
    
    def _solve_twice(self, b, bset):

        # Solve (L+beta*I) X2 = B
        self.chol.solve2(
            b, 
            bset, 
            self.X2, 
            self.Xset, 
            self.Y, 
            self.E
        ) 

        # Solve (L+beta*I) X1 = X2
        self.chol.solve2(
            self.X2, 
            None, 
            self.X1, 
            self.Xset, 
            self.Y, 
            self.E
        ) 

        return self.X1

    def _mult(self, x, scalar):
        
        # x = beta * x
        self.chol.sdmult(
            x,  # n/a
            x,  # beta  * x 
            alpha = 0.0, 
            beta  = scalar
        )

        return x
    
    def _mult_lap(self, x, scalar):
        
        # x = alpha * L * x
        self.chol.sdmult(
            matrix_ptr = x, 
            out_ptr = self.X2,  
            alpha = scalar, 
            beta  = 0.0
        )
        return self.X2

    
    def wavelet_coeffs2(self, f, fset=None, scales=None):
        '''
        Returns
            Wavelet  coeffs of indicated scale using the analytical form.
            (1/s)  L/(L+I/s)^2  located only at a subset of buses
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate wavelet coeffs.
            fset: (nVerticies x 1) Sparse vector indicator function of nodes 
                where the wavelet coeffs need to be solved. Much faster than calculating
                coefficients for every vertex localization. Default: None, does not consider fset.
            scales: list (numScales) of scales to compute wavelet coefficents for.
        Returns
            Wavelet coefficients for each scale (numVerticies x numScales)
            Solved accurately only for buses indicated by fset
        '''

        # Scales
        scales = self.scales if scales is None else scales
  
        # Pointer to f (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(f))
        Bset = None

        # Using this requires the number of columns in f to be 1
        if fset is not None:
            Bset = byref(self.chol.numpy_to_chol_sparse_vec(fset))

        # My workspace pointers
        X1, X2   = self.X1, self.X2
        Xset     = self.Xset
        Y , E    = self.Y , self.E

        # Store Resulting matricies here
        W = []

        # For each wavelet scale
        for i, scale in enumerate(self.scales):

            # Step 1 -> Set Scale via shifted numeric factor
            self.chol.num_factor(1/scale)

            # Step 2 -> Solve (L+beta*I) B = X1
            self.chol.solve2(B, Bset, X1, Xset, Y, E) 

            # Step 3 -> Solve (L+beta*I) X1 = X2
            self.chol.solve2(X1, None, X2, Xset, Y, E) 
       
            # Step 4 -> Compute X1 = (4/scale)Lap@X2 
            self.chol.sdmult(X2, X1, 4/scale)

            # Convert to numpy (copy)
            x_numpy = self.chol.chol_dense_to_numpy(X1)
            
            # Append to list
            W.append(x_numpy)

        # Although returns full matrix, only rows specified by Bset are valid
        return W
       
    def highpass_coeffs3(self, f, fset=None, scales=None):
        '''
        Description
            Scaling coefficnets at indicated scales using the analytical form
            aL/(aL+I)
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate HP coeffs.
            fset: Pattern vector 
            scales: list (numScales) of scales to compute  HP coefficents for.
        Returns
            High-pass coefficients for each scale (numVerticies x numScales)
        '''

        # Scales
        scales = self.scales if scales is None else scales
  
        # Pointer to f (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(f))
        Bset = None

        # Using this requires the number of columns in f to be 1
        if fset is not None:
            Bset = byref(self.chol.numpy_to_chol_sparse_vec(fset))

        # My workspace pointers
        X1, X2   = self.X1, self.X2
        Xset     = self.Xset
        Y , E    = self.Y , self.E

        # Store Resulting matricies here
        W = []



        # For each wavelet scale
        for i, scale in enumerate(self.scales):

            # Step 1 -> Set Scale via shifted numeric factor
            self.chol.num_factor(1/scale)

            if i==0:
                # Useless solve to init X2
                self.chol.solve2(B, Bset, X2, Xset, Y, E) 


            # Step 2 -> Solve (L+beta*I) B = X1
            self.chol.solve2(B, Bset, X1, Xset, Y, E) 

            # Step 3 -> Compute L@X1 
            self.chol.sdmult(X1, X2, 1.0)
     
            # Convert to numpy (copy)
            x_numpy = self.chol.chol_dense_to_numpy(X2)
            
            # Append to list
            W.append(x_numpy)

        # Although returns full matrix, only rows specified by Bset are valid
        return W