from .structs import *
from ctypes import byref, cast, POINTER, CDLL

import numpy as np
from importlib_resources import files, as_file

CHOLMOD_A   = 0 #  0  /* solve Ax=b    */
#define CHOLMOD_LDLt 1  /* solve LDL'x=b */
#define CHOLMOD_LD   2  /* solve LDx=b   */
#define CHOLMOD_DLt  3  /* solve DL'x=b  */
#define CHOLMOD_L    4  /* solve Lx=b    */
#define CHOLMOD_Lt   5  /* solve L'x=b   */
#define CHOLMOD_D    6  /* solve Dx=b    */
#define CHOLMOD_P    7  /* permute x=Px  */
#define CHOLMOD_Pt   8  /* permute x=P'x */



class CholWrapper:

    def __init__(self, A) -> None:
        ''' 
        dll: the ctype dll compiles CHOLMOD 
        A: csc_matrix - the working matrix, which can undergo lowrank changes
        '''

        # Access DLL
        with as_file(
            files("sgwt").joinpath("cholesky/cholmod.dll")
        ) as dll_path:
            self.dll = CDLL(str(dll_path))


        # Parse matrix to cholmod_sparse
        self.A = self.to_cholmod_sparse(A) # Parse to Cholmod format
        self.common = cholmod_common()

        # Syntax Sugar Pointer Names
        self.A_ptr = byref(self.A)
        self.common_ptr = byref(self.common)

        # DLL Setup    
        self.config_function_args(self.dll)
        self.config_return_types(self.dll)

        # TODO Support other solve types
        self.MODE = CHOLMOD_A

        # Start Cholesky by default
        self.start()

    def status(self):
        ''' 
        Cholmod Status: 
        0 OK; 
        -4 Invalid Input; 
        -2 Out of Mem
        '''
        stat = self.common.status
        print("CHOLMOD status:", stat)
        return stat
    
    def to_cholmod_sparse(self, A, itype=0, dtype=0):
        """
        Convert a 2D NumPy array A into a cholmod_sparse struct.
        
        Parameters:
            A : np.ndarray
                Dense or 2D array to convert.
            itype : int
                0=int32 indices, 1=int64 indices
            dtype : int
                0=double (real), 1=float (single), 2=complex
        
        Returns:
            cholmod_sparse instance.
        """
        # Convert to CSC format
        nrow, ncol = A.shape
        
        # Prepare contiguous arrays
        x = np.ascontiguousarray(A.data, dtype=np.float64)
        i = np.ascontiguousarray(A.indices, dtype=np.int32 if itype==0 else np.int64)
        p = np.ascontiguousarray(A.indptr, dtype=np.int32 if itype==0 else np.int64)
        
        # Cast to void pointers for ctypes
        x_ptr = x.ctypes.data_as(c_void_p)
        i_ptr = i.ctypes.data_as(c_void_p)
        p_ptr = p.ctypes.data_as(c_void_p)
        
        # Initialize struct
        cholA = cholmod_sparse()
        cholA.nrow = nrow
        cholA.ncol = ncol
        cholA.nzmax = len(x)
        cholA.p = p_ptr
        cholA.i = i_ptr
        cholA.x = x_ptr
        cholA.z = None             # None if real
        cholA.stype = 1            # 0 = general, 1 = symmetric store upper part
        cholA.itype = itype
        cholA.xtype = 1            # 1 = real
        cholA.dtype = dtype
        cholA.sorted = 1           # columns sorted
        cholA.packed = 1           # packed
        
        return cholA
    
    # TODO improve, slow
    def to_dense_vector(self, b: np.ndarray):
        if not isinstance(b, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")

        # Ensure correct dtype
        #if b.dtype != np.float64:
        #    b = b.astype(np.float64, copy=False)

        # Ensure contiguous memory
        #if not b.flags["C_CONTIGUOUS"]:
        #    b = np.ascontiguousarray(b)
        #b = np.asfortranarray(b)

        # Ensure 1D
        print(b.ndim)
        if b.ndim != 2:
            raise ValueError("values must be a 2D array")

        n = b.size
        '''
        # Allocate CHOLMOD dense
        D = self.dll.cholmod_allocate_dense(
            c_size_t(n), c_size_t(1), c_size_t(n), c_int(1), 
            self.common_ptr
        )
        '''
        D = cholmod_dense()
        D.nrow = b.shape[0]
        D.ncol = b.shape[1]
        D.nzmax = b.size
        D.d = b.shape[0]
        D.x = b.ctypes.data_as(c_void_p)
        D.xtype = 1 # real
        D.dtype = 0 # c_double, real

        
        # if not D:
        #    raise MemoryError("CHOLMOD allocate dense failed")

        # Zero-copy pointer assignment
        #D.contents.x = b.ctypes.data_as(c_void_p)

        # CRITICAL: keep NumPy array alive to prevent segfault
        #self._dense_ref = b

        return D
    
    def start(self):
        '''
        Starts cholmod.
        '''
        self.dll.cholmod_start(byref(self.common))

    def finish(self):
        '''
        Finish the cholmod usage. 
        TODO determine when to call, uncertain for python implementation.
        '''
        # Free Factored Object
        self.dll.cholmod_free_factor(self.fact, self.common_ptr)

        # TODO FREE MATRIX AND DENSE
        #libcholmod.cholmod_free_sparse(byref(A), byref(common))

        # WARNING make sure this does not conflict with Numpy memory. 
        # Don't think it will
        #libcholmod.cholmod_free_dense(byref(b), byref(common))
        self.dll.cholmod_finish(byref(self.common))


    def sym_factor(self):
        ''' 
        Performs symbolic factorization using cholmod_analyze
        '''
        self.fact = self.dll.cholmod_analyze(self.A_ptr, self.common_ptr)
        self.fact_ptr = byref(self.fact)
        
    def num_factor(self, beta):
        ''' 
        Description
            Equivilent to choldmod_factorize_p in CHOLMOD.
            The matrix is assumed to be the same that underwent symbolic factorization.
        Parameters
            beta: real number (for the GSP application, must be positive)
        '''

        # Must be complex for DLL use
        beta_cmplx = (c_double * 2)(beta, 0.0) 

        self.dll.cholmod_factorize_p(
            self.A_ptr,
            beta_cmplx,
            None, # fset
            0,    # fisze
            self.fact,
            self.common_ptr
        )


    def solve(self, b):
        '''
        Description
            Equivilent to choldmod_spsolve in CHOLMOD
        Parameters
            b: (N, M) 2D numpy array exlusively, for now 
        '''

        x_ptr = self.dll.cholmod_solve(
            self.MODE, 
            self.fact, 
            self.to_dense_vector(b), # TODO improve, slow
            self.common_ptr
        )

        self.status()


        # TODO we can support sparse input B
        # if we use cholmod_spsolve
        # But for a general signal b will be dense
        # it returns choldmod_sparse so I need to
        # change how I read the results here I read dense result)

        # TODO okay Technically I should clear memory
        nsol = x_ptr.contents.nrow
        sol = np.ctypeslib.as_array(cast(x_ptr.contents.x, POINTER(c_double)), shape=(nsol,1))

        return sol

    
    def config_function_args(self, dll):

        dll.cholmod_start.argtypes = [POINTER(cholmod_common)]
        dll.cholmod_finish.argtypes = [POINTER(cholmod_common)]

        dll.cholmod_allocate_sparse.argtypes = [
            c_size_t, c_size_t, c_size_t,
            c_int, c_int, c_int, c_int,
            POINTER(cholmod_common)
        ]

        # Symbolic Factorization
        dll.cholmod_analyze.argtypes = [
            POINTER(cholmod_sparse),
            POINTER(cholmod_common)
        ]

        # Numeric factorization w/ Shifting
        dll.cholmod_factorize_p.argtypes = [
            POINTER(cholmod_sparse),          # A
            POINTER(c_double),                # beta[2]
            POINTER(c_int32),                  # fset (int32_t*)
            c_size_t,                          # fsize
            POINTER(cholmod_factor),           # L
            POINTER(cholmod_common)            # Common
        ]

        # For a general 'b' vector, lots of data
        dll.cholmod_solve.argtypes = [
            c_int,
            POINTER(cholmod_factor),
            POINTER(cholmod_dense),
            POINTER(cholmod_common)
        ]

        # For sparse 'b' vector, like an impulse
        dll.cholmod_spsolve.argtypes = [
            c_int,
            POINTER(cholmod_factor),
            POINTER(cholmod_dense),
            POINTER(cholmod_common)
        ]


        dll.cholmod_allocate_dense.argtypes = [
            c_size_t, c_size_t, c_size_t, c_int,
            POINTER(cholmod_common)
        ]
        dll.cholmod_free_sparse.argtypes = [
            POINTER(POINTER(cholmod_sparse)),
            POINTER(cholmod_common)
        ]
        dll.cholmod_free_dense.argtypes = [
            POINTER(POINTER(cholmod_dense)),
            POINTER(cholmod_common)
        ]
        dll.cholmod_free_factor.argtypes = [
            POINTER(POINTER(cholmod_factor)),
            POINTER(cholmod_common)
        ]
        dll.cholmod_norm_sparse.argtypes = [
            POINTER(cholmod_sparse),  # A
            c_int,                    # norm type: 0=inf, 1=1
            POINTER(cholmod_common)   # Common
        ]

    def config_return_types(self, dll):
        dll.cholmod_start.restype = None
        dll.cholmod_finish.restype = None
        dll.cholmod_allocate_sparse.restype = POINTER(cholmod_sparse)

        dll.cholmod_analyze.restype = POINTER(cholmod_factor)
        dll.cholmod_factorize_p.restype = c_int
        dll.cholmod_solve.restype = POINTER(cholmod_dense)
        dll.cholmod_spsolve.restype = POINTER(cholmod_sparse)

        dll.cholmod_allocate_dense.restype = POINTER(cholmod_dense)
        dll.cholmod_free_sparse.restype = None
        dll.cholmod_free_dense.restype = None
        dll.cholmod_free_factor.restype = None
        dll.cholmod_norm_sparse.restype = c_double


    


