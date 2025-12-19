

from .analytic import AnalyticFilters

from ..cholesky import CholWrapper, cholmod_dense, cholmod_sparse
from ctypes import byref,  POINTER


class FiltersDLL(AnalyticFilters):
    '''
    A sparse memory efficient implementation
    that uses cholmod_solve2
    '''

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
    Abstract Function Definitions
    '''

    def _allocate_results(self, b, scales):
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

    def _symbolic_factorization(self, L):
        self.chol = CholWrapper(L)
    
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
        self.chol.sdmult(
            matrix_ptr = x, 
            out_ptr = self.X2,  
            alpha = scalar, 
            beta  = 0.0
        )
        return self.X2

    def _save_to_results(self, x, index):
        self.results.append(
            self.chol.chol_dense_to_numpy(x)
        )
