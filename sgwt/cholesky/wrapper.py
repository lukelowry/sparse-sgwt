# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/cholesky/wrapper.py
Description: Low-level Python wrapper for the CHOLMOD C library.
"""

from ctypes import byref, cast, POINTER, c_int32
import numpy as np

from .structs import *
from ..io import get_cholmod_dll

# Numeric precision
CHOLMOD_SINGLE = 0   # 32-bit float
CHOLMOD_DOUBLE = 1   # 64-bit float

# Matrix value type
CHOLMOD_PATTERN = 0  # structure only
CHOLMOD_REAL    = 1  # real
CHOLMOD_COMPLEX = 2  # complex (interleaved)
CHOLMOD_ZOMPLEX = 3  # complex (split)

# Sparse format
CHOLMOD_TRIPLET = 0  # triplet form
CHOLMOD_SPARSE  = 1  # CSC

# Factor representation
CHOLMOD_SIMPLICIAL = 0  # simplicial
CHOLMOD_SUPERNODAL = 1  # supernodal

# Factor form
CHOLMOD_L  = 0  # LLᵀ
CHOLMOD_LT = 1  # LDLᵀ

# Up/down-date
CHOLMOD_UPDATE   = 1  # update
CHOLMOD_DOWNDATE = 0  # downdate

# Solve selectors
CHOLMOD_A  = 0  # Ax=b
CHOLMOD_L  = 1  # Lx=b
CHOLMOD_LT = 2  # Lᵀx=b
CHOLMOD_D  = 3  # Dx=b
CHOLMOD_P  = 4  # permutation

# Ordering
CHOLMOD_NATURAL = 0  # natural
CHOLMOD_GIVEN   = 1  # given
CHOLMOD_AMD     = 2  # AMD
CHOLMOD_METIS   = 3  # METIS
CHOLMOD_NESDIS  = 4  # nested dissection

# Booleans
CHOLMOD_FALSE = 0
CHOLMOD_TRUE  = 1

class CholWrapper:
    """
    A wrapper class for interacting with CHOLMOD DLL

    WARNING: Should only be used indirectly through SGWT Object
    otherwise memory leaks may occur.
    """

    def __init__(self, A) -> None:
        """Initializes the CHOLMOD wrapper.

        Parameters
        ----------
        A : scipy.sparse.csc_matrix
            The sparse matrix (e.g., Graph Laplacian) to be analyzed.
            It is converted to an internal CHOLMOD sparse format.
        """
        self.dll = get_cholmod_dll()
        
        # DLL Setup    
        self.config_function_args(self.dll)
        self.config_return_types(self.dll)

        # Parse matrix to cholmod_sparse
        self.A = self.numpy_to_chol_sparse(A) # Parse to Cholmod format
        
        # Make choldmod_common struct
        self.common = cholmod_common()

        # TODO Support other solve types
        self.MODE = CHOLMOD_A

    def status(self):
        """
        Returns the status of the CHOLMOD common object.

        Returns
        -------
        int
             0: OK, -4: Invalid Input, -2: Out of Memory
        """
        return self.common.status

    # --------------------------------------------------------------------------
    # Factorizations
    # --------------------------------------------------------------------------

    def sym_factor(self):
        """
        Performs symbolic factorization using cholmod_analyze.
        """
        self.fact_ptr = self.dll.cholmod_analyze(
            byref(self.A),  
            byref(self.common)
        )
        
    def num_factor(self, A_ptr, fact_ptr, beta):
        """
        Performs numeric factorization with shifting (cholmod_factorize_p).
        
        The matrix is assumed to be the same that underwent symbolic factorization.

        Parameters
        ----------
        A_ptr : POINTER(cholmod_sparse)
            Matrix to factor.
        fact_ptr : POINTER(cholmod_factor)
            Factorization structure (input/output).
        beta : float
            Shift parameter (A + beta*I). For GSP, must be positive.
        """
        # Must be complex for DLL use
        beta_cmplx = (c_double * 2)(beta, 0.0) 

        self.dll.cholmod_factorize_p(
            A_ptr,       # Matrix to factor
            beta_cmplx,
            None,        # fset
            0,           # fsize
            fact_ptr,    # (In/Out)
            byref(self.common)
        )

    # --------------------------------------------------------------------------
    # Solving
    # --------------------------------------------------------------------------

    def solve2(self, fact_ptr, B_ptr, Bset_ptr, X_ptr, Xset_ptr, Y_ptr, E_ptr):
        """
        Solves the linear system using cholmod_solve2.

        Parameters
        ----------
        fact_ptr : POINTER(cholmod_factor)
            Factorized matrix L.
        B_ptr : POINTER(cholmod_dense)
            Right-hand side matrix B.
        Bset_ptr : POINTER(cholmod_sparse)
            Sparse subset of B (optional).
        X_ptr : POINTER(cholmod_dense)
            Solution matrix X (output).
        Xset_ptr : POINTER(cholmod_sparse)
            Sparse subset of X (output).
        Y_ptr : POINTER(cholmod_dense)
            Workspace.
        E_ptr : POINTER(cholmod_dense)
            Workspace.

        Returns
        -------
        int
            1 if successful, 0 otherwise.
        """
        return self.dll.cholmod_solve2(
            self.MODE,        # (In ) int ---- Ax=b
            fact_ptr,         # (In ) chol_factor *L 
            B_ptr,            # (In ) chol_dense  *B 
            Bset_ptr,         # (In ) chol_sparse *Bset 
            byref(X_ptr),     # (Out) cholmod_dense **X_Handle (where sol is stored)
            byref(Xset_ptr),  # (Out) cholmod_sparse **Xset_Handle, byref(Xset_ptr)
            byref(Y_ptr),     # (Workspace)  **Y
            byref(E_ptr),     # (Workspace) **E
            byref(self.common)
        )

    def solve(self, fact_ptr, b_ptr):
        """
        Solves the linear system using cholmod_solve.

        Parameters
        ----------
        fact_ptr : POINTER(cholmod_factor)
            Factorized matrix L.
        b_ptr : POINTER(cholmod_dense)
            Right-hand side vector/matrix b.

        Returns
        -------
        POINTER(cholmod_dense)
            Solution vector/matrix x.
        """
        return self.dll.cholmod_solve(
            self.MODE, 
            fact_ptr, 
            b_ptr, 
            byref(self.common)
        )

    # --------------------------------------------------------------------------
    # Matrix Operations
    # --------------------------------------------------------------------------

    def sdmult(self, matrix_ptr, out_ptr, alpha=1.0, beta=0.0):
        """
        Sparse matrix multiplication: out = alpha * (A @ matrix) + beta * matrix.

        Parameters
        ----------
        matrix_ptr : POINTER(cholmod_dense)
            Input dense matrix.
        out_ptr : POINTER(cholmod_dense)
            Output dense matrix.
        alpha : float, optional
            Scaling factor for A @ matrix.
        beta : float, optional
            Scaling factor for matrix.
        """
        Alpha = (c_double * 2)(alpha, 0.0) 
        Beta  = (c_double * 2)(beta, 0.0) 

        self.dll.cholmod_sdmult(
            byref(self.A),  # Left matrix always Laplacian
            0,              # Do not Transpose = 0
            Alpha,          # out += Alpha * (Lap @ matrix)
            Beta,           # out += Beta * matrix
            matrix_ptr,     # Input
            out_ptr,        # Output
            byref(self.common) 
        )

    # --------------------------------------------------------------------------
    # Low Rank Updates
    # --------------------------------------------------------------------------

    def submatrix(self, A_ptr, rset, rsize, cset=None, csize=-1, mode=1, sorted=1):
        """
        Extracts a submatrix from a cholmod_sparse matrix.

        This is a wrapper for `cholmod_submatrix`.

        Parameters
        ----------
        A_ptr : POINTER(cholmod_sparse)
            Pointer to the sparse matrix to extract from.
        rset : POINTER(c_int32)
            Set of row indices to extract.
        rsize : int
            Number of rows to extract.
        cset : POINTER(c_int32), optional
            Set of column indices to extract (default is None).
        csize : int, optional
            Number of columns to extract (default is -1 for all columns).
        mode : int, optional
            Extraction mode (default is 1):
            - 0: pattern
            - 1: numerical (non-conj.)
            - 2: numerical (conj)
        sorted : int, optional
            Whether the input index sets are sorted (default is 1).

        Returns
        -------
        POINTER(cholmod_sparse)
            A pointer to the newly created sparse submatrix.
        """
        return self.dll.cholmod_submatrix(
            A_ptr,                 # Ptr to sparse Matrix
            rset,                  # rset (int32_t*)
            rsize,                 # rsize of rset, or -1 for ":"
            cset,                  # cset (int32_t*)
            csize,                 # size of cset, or -1 for ":"
            mode,                  # mode (2, 1, 0)
            sorted,                # sorted = True (1, 0)
            byref(self.common),   # Common
        )
    
    def _permute_sparse(self, C_ptr):
        """
        Returns the permuted C matrix by L->Perm.
        """
        L_Perm = cast(self.fact_ptr.contents.Perm, POINTER(c_int32))
        L_n    = self.fact_ptr.contents.n

        # Per Documentation of updown We must manually permute C by the optimal fill ordering
        #  Cnew = cholmod_submatrix (C, L->Perm, L->n, NULL, -1, TRUE, TRUE, Common)
        Cnew = self.submatrix(C_ptr, L_Perm, L_n, None, -1, 1, 1)

        # Ensure not null
        if not bool(Cnew):
            print("Submatrix is NULL! Updown will have no effect.")

        return Cnew

    def updown(self, update, C_ptr, fact_ptr):
        """
        Updates or downdates the factorization (LDL' +/- CC').

        Parameters
        ----------
        update : int
            1 for update, 0 for downdate.
        C_ptr : POINTER(cholmod_sparse)
            Column vector/matrix for the update.
        fact_ptr : POINTER(cholmod_factor)
            Factorization to modify.

        Returns
        -------
        int
            1 if successful, 0 otherwise.
        """
        
        # Permute by L->Perm
        Cnew = self._permute_sparse(C_ptr)

        # Perform updown
        ok = self.dll.cholmod_updown(
            update,             # (1 update, 0 downdate)
            Cnew,               # Pointer to sparse incoming update
            fact_ptr,      # Pointer to existing factorization
            byref(self.common)  # Pointer to common
        )

        # Free the permuted matrix
        self.free_sparse(Cnew)

        return ok
    
    def updown_solve(self, update, C_ptr, fact_ptr, X_ptr, deltaB_ptr):
        """Solves a system after a rank-k update/downdate.

        This is a wrapper for `cholmod_updown_solve`. It is more efficient
        than calling `updown` followed by `solve`.

        Parameters
        ----------
        update : int
            1 for update, 0 for downdate.
        C_ptr : POINTER(cholmod_sparse)
            Column vector/matrix for the update.
        fact_ptr : POINTER(cholmod_factor)
            Factorization to modify.
        X_ptr : POINTER(cholmod_dense)
            Solution to the original system.
        deltaB_ptr : POINTER(cholmod_dense)
            Change in the right-hand-side B.

        Returns
        -------
        int
            1 if successful, 0 otherwise.
        """
        # Permute by L->Perm
        Cnew = self._permute_sparse(C_ptr)

        ok = self.dll.cholmod_updown_solve(
            update,         # (Update, Downdate )(1,0)
            Cnew,           # The permuted update sparse matrix pointer 
            fact_ptr,  # factor
            X_ptr,          # Solution
            deltaB_ptr,     # Deviated input?
            byref(self.common)

        )

        self.free_sparse(Cnew)

        return ok
    

    # --------------------------------------------------------------------------
    # Low Rank Updates Syntax Sugar
    # --------------------------------------------------------------------------
    
    def update(self, C_ptr, fact_ptr):
        """
        Calculates new L (factorization) for A + CC^T.
        
        LDL' = P(A + CC^T)P^T

        Parameters
        ----------
        C_ptr : POINTER(cholmod_sparse)
            Update matrix.
        fact_ptr : POINTER(cholmod_factor)
            Factorization to update.
        """
        return self.updown(True, C_ptr, fact_ptr)
        
    def downdate(self, C_ptr, fact_ptr):
        """
        Calculates new L (factorization) for A - CC^T.
        
        LDL' = P(A - CC^T)P^T

        Parameters
        ----------
        C_ptr : POINTER(cholmod_sparse)
            Downdate matrix.
        fact_ptr : POINTER(cholmod_factor)
            Factorization to downdate.
        """
        return self.updown(False, C_ptr, fact_ptr)
    
    # --------------------------------------------------------------------------
    # Data Structures
    # --------------------------------------------------------------------------
        
    def numpy_to_chol_sparse(self, A, itype=0, dtype=0) -> cholmod_sparse:
        """
        Convert a 2D NumPy array A into a cholmod_sparse struct.
        
        Parameters
        ----------
        A : csc_matrix
            Sparse vector/matrix to convert.
        itype : int, optional
            0=int32 indices, 1=int64 indices.
        dtype : int, optional
            0=double (real), 1=float (single), 2=complex.
        
        Returns
        -------
            cholmod_sparse instance (sorted, packed)
        """

        #  Get Shape
        nrow, ncol = A.shape
        
        # Prepare contiguous arrays
        x = np.asfortranarray(A.data, dtype=np.float64)
        i = np.asfortranarray(A.indices, dtype=np.int32 if itype==0 else np.int64)
        p = np.asfortranarray(A.indptr, dtype=np.int32 if itype==0 else np.int64)
        
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
        cholA.sorted = 1           # Sorted = True
        cholA.packed = 1           # packed = True
        
        return cholA
    
    def numpy_to_chol_sparse_vec(self, A, itype=0, dtype=0):
        """
        Converts a scipy.sparse matrix to a cholmod_sparse struct for Bset.

        Parameters
        ----------
        A : csc_matrix
            Input sparse matrix.
        itype : int, optional
            Index type (0: int32, 1: int64).
        dtype : int, optional
            Data type (0: double, 1: single).

        Returns
        -------
        cholmod_sparse
            CHOLMOD sparse structure.
        """
        #  Get Shape
        nrow, ncol = A.shape
        
        # Prepare contiguous arrays
        x = np.asfortranarray(A.data, dtype=np.float64)
        i = np.asfortranarray(A.indices, dtype=np.int32 if itype==0 else np.int64)
        p = np.asfortranarray(A.indptr, dtype=np.int32 if itype==0 else np.int64)
        
        # Cast to void pointers for ctypes
        x_ptr = x.ctypes.data_as(c_void_p)
        i_ptr = i.ctypes.data_as(c_void_p)
        p_ptr = p.ctypes.data_as(c_void_p)
        
        # Initialize struct
        bset = cholmod_sparse()
        bset.nrow = nrow
        bset.ncol = ncol
        bset.nzmax = len(x)
        bset.p = p_ptr
        bset.i = i_ptr
        bset.x = x_ptr
        bset.z = None             # None if real
        bset.stype = 0            # 0 = general, 1 = symmetric store upper part
        bset.itype = itype
        bset.xtype = 0            # 0 = pattern, 1 = real 
        bset.dtype = dtype
        bset.sorted = 0           # Sorted = NOTE FALSE for Bset
        bset.packed = 1           
        
        return bset
    
    def numpy_to_chol_dense(self, b: np.ndarray) -> cholmod_dense:
        """
        Converts a NumPy array to a cholmod_dense struct.

        Parameters
        ----------
        b : np.ndarray
            Input array (must be 2D and Fortran contiguous).

        Returns
        -------
        cholmod_dense
            CHOLMOD dense structure.
        """

        if not isinstance(b, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")

        # Ensure correct dtype
        if b.dtype != np.float64:
            b = b.astype(np.float64, copy=False)

        # Ensure contiguous memory
        if not b.flags["F_CONTIGUOUS"]:
            raise ValueError("b must be Fortran-contiguous for zero-copy CHOLMOD dense")

        # Ensure 2D
        if b.ndim != 2:
            raise ValueError("values must be a 2D array")

        # TODO use new constructor

        # Zero Copy into CHOLMOD dense format
        D = cholmod_dense()
        D.nrow = b.shape[0] # Row Size
        D.ncol = b.shape[1] # Column Size
        D.nzmax = b.size    # Max Count of Non-Zero Elements
        D.d = b.shape[0]    # Leading Dimension
        D.x = b.ctypes.data_as(c_void_p) # Pointer to numpy memory
        D.xtype = 1         # real
        D.dtype = 0         # c_double, real

        # Return ctype.Structure
        return D

    def chol_dense_to_numpy(self, x_ptr):
        """
        Converts a cholmod_dense pointer to a NumPy array.

        Parameters
        ----------
        x_ptr : POINTER(cholmod_dense)
            Pointer to the dense matrix.

        Returns
        -------
        np.ndarray
            Copy of the data as a NumPy array.
        """

        # Create a View
        nrow = x_ptr.contents.nrow
        ncol = x_ptr.contents.ncol
        d    = x_ptr.contents.d
        buf = cast(x_ptr.contents.x, POINTER(c_double))
        
        # NOTE The order 'F' is crucial for correct reading of memory
        # CHOLMOD stores in fortran order (col-major)
        # Numpy stores C-Order (row-major)
        x_view = np.ndarray(
            shape=(nrow, ncol),
            dtype=np.float64,
            buffer=np.ctypeslib.as_array(buf, shape=(d * ncol,)),
            order="F",
        )


        # Copy Cholmod Mem (Must still be freed)
        return x_view.copy(order='F')

    def triplet_to_chol_sparse(self,nrow, ncol, rows, cols, vals, stype=0):
        """
        Create a CHOLMOD sparse matrix from triplet form (rows, cols, vals).

        Parameters
        ----------
        nrow : int
            Number of rows.
        ncol : int
            Number of columns.
        rows : list
            Row indices (0-based).
        cols : list
            Column indices (0-based).
        vals : list
            Values.
        stype : int, optional
            Symmetry flag (0=unsymmetric, >0=upper, <0=lower).

        Returns
        -------
        POINTER(cholmod_sparse)
            Pointer to the allocated matrix.
        """
        nzmax = len(vals)
        
        # Allocate CHOLMOD sparse matrix
        Cptr = self.dll.cholmod_allocate_sparse(
            c_size_t(nrow),
            c_size_t(ncol),
            c_size_t(nzmax),
            c_int(1),         # sorted
            c_int(1),         # packed
            c_int(stype),
            c_int(1),         # numeric type: double
            byref(self.common)
        )
        
        # Access internal arrays
        i_array = cast(Cptr.contents.i, POINTER(c_int))
        x_array = cast(Cptr.contents.x, POINTER(c_double))
        p_array = cast(Cptr.contents.p, POINTER(c_int))
        
        # Count nonzeros per column
        col_counts = [0]*ncol
        for c in cols:
            col_counts[c] += 1
        
        # Compute column pointers (cumulative sum)
        p_array[0] = 0
        for j in range(1, ncol+1):
            p_array[j] = p_array[j-1] + col_counts[j-1]
        
        # Track current insertion position per column
        next_pos = [p_array[j] for j in range(ncol)]
        
        # Fill row indices and values
        for k in range(nzmax):
            col = cols[k]
            pos = next_pos[col]
            i_array[pos] = rows[k]
            x_array[pos] = vals[k]
            next_pos[col] += 1

        return Cptr

    # --------------------------------------------------------------------------
    # Cholmod Context
    # --------------------------------------------------------------------------

    def start(self):
        """
        Starts CHOLMOD (cholmod_start).
        """
        self.dll.cholmod_start(
            byref(self.common)
        )

    def finish(self):
        """
        Finishes CHOLMOD (cholmod_finish).
        """

        self.dll.cholmod_finish(
            byref(self.common)
        )

    # --------------------------------------------------------------------------
    # Freeing Memory
    # --------------------------------------------------------------------------

    def free_factor(self, fact_ptr):
        """
        Frees a cholmod_factor struct.

        Parameters
        ----------
        fact_ptr : POINTER(cholmod_factor)
            Factor to free.
        """
        self.dll.cholmod_free_factor(
            fact_ptr, 
            byref(self.common)
        )

    def free_dense(self, dense_ptr):
        """
        Frees a cholmod_dense struct.

        Parameters
        ----------
        dense_ptr : POINTER(cholmod_dense)
            Dense matrix to free.
        """
        self.dll.cholmod_free_dense(
            dense_ptr, 
            byref(self.common)
        )

    def free_sparse(self, sparse_ptr):
        """
        Frees a cholmod_sparse struct.

        Parameters
        ----------
        sparse_ptr : POINTER(cholmod_sparse)
            Sparse matrix to free.
        """
        self.dll.cholmod_free_sparse(
            sparse_ptr, 
            byref(self.common)
        )


    # --------------------------------------------------------------------------
    # Allocating Memory
    # --------------------------------------------------------------------------

    def allocate_dense(self, nrow, ncol):
        """Allocates a dense matrix in CHOLMOD.

        This is a wrapper for `cholmod_allocate_dense`. The memory is managed
        by CHOLMOD and must be freed with `free_dense`.

        Parameters
        ----------
        nrow : int
            Number of rows.
        ncol : int
            Number of columns.

        Returns
        -------
        POINTER(cholmod_dense)
            A pointer to the newly allocated dense matrix.
        """
        return self.dll.cholmod_allocate_dense(
            nrow,
            ncol,
            nrow,
            1, # real?
            byref(self.common)
        )
    
    def allocate_sparse_matrix(dll, nrow, ncol, nzmax, stype, sorted=True, packed=True, common=None):
        """
        Allocate a CHOLMOD sparse matrix entirely in CHOLMOD memory.
        Returns POINTER(cholmod_sparse).
        """
        sorted_flag = 1 if sorted else 0
        packed_flag = 1 if packed else 0
        return dll.cholmod_allocate_sparse(
            c_size_t(nrow),
            c_size_t(ncol),
            c_size_t(nzmax),
            c_int(sorted_flag),
            c_int(packed_flag),
            c_int(stype),
            c_int(1),            # numeric type: double
            byref(common)
        )
    
    def alloc_factor(self, n, dtype):
        """
        Allocate an empty cholmod_factor structure.

        Parameters
        ----------
        n : int
            Dimension of the n-by-n matrix to be factorized.
        dtype : int
            CHOLMOD_SINGLE or CHOLMOD_DOUBLE.

        Returns
        -------
        L_ptr : POINTER(cholmod_factor)
            Allocated factor object. Symbolic and numeric contents are uninitialized.
        """

        L_ptr = self.cholmod.cholmod_alloc_factor(
            n,
            dtype,
            self.common
        )

        if not L_ptr:
            raise RuntimeError("cholmod_alloc_factor failed")

        return L_ptr
    
    def zeros(self, nrow, ncol):
        """Creates a dense matrix of zeros in CHOLMOD.

        This is a wrapper for `cholmod_zeros`. The memory is managed
        by CHOLMOD and must be freed with `free_dense`.

        Parameters
        ----------
        nrow : int
            Number of rows.
        ncol : int
            Number of columns.

        Returns
        -------
        POINTER(cholmod_dense)
            A pointer to the newly allocated dense matrix filled with zeros.
        """
        return self.dll.cholmod_zeros(
            nrow,
            ncol,
            1, # real?
            byref(self.common)
        )
    
    # --------------------------------------------------------------------------
    # Copying
    # --------------------------------------------------------------------------

    def copy_factor(self, L_ptr):
        """
        Create a deep copy of a cholmod_factor.

        Parameters
        ----------
        L_ptr : POINTER(cholmod_factor)
            Existing factor to copy. Not modified.

        Returns
        -------
        L_copy_ptr : POINTER(cholmod_factor)
            Independent copy of the factor.
        """

        if not L_ptr:
            raise ValueError("L_ptr is NULL")

        L_copy_ptr = self.dll.cholmod_copy_factor(
            L_ptr,
            self.common
        )

        if not L_copy_ptr:
            raise RuntimeError("cholmod_copy_factor failed")

        return L_copy_ptr

    # --------------------------------------------------------------------------
    # Configuration Functions
    # --------------------------------------------------------------------------

    def config_function_args(self, dll):

        # Symbolic Factorization
        dll.cholmod_analyze.argtypes = [
            POINTER(cholmod_sparse),           # A: Matrix to analyze
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

        # Numeric factorization w/ Shifting
        dll.cholmod_factorize_p.argtypes = [
            POINTER(cholmod_sparse),           # A: Matrix to factor
            POINTER(c_double),                 # beta: Shift (beta[2])
            POINTER(c_int32),                  # fset: Subset of rows/cols
            c_size_t,                          # fsize: Size of fset
            POINTER(cholmod_factor),           # L: Factorization
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

        # Reused workspace and specified locality/sparisty
        # best for subset of wavelet coefficients
        dll.cholmod_solve2.argtypes = [
            c_int,                             # sys: System to solve (Ax=b, Lx=b, etc.)
            POINTER(cholmod_factor),           # L: Factorization
            POINTER(cholmod_dense),            # B: Right hand side
            POINTER(cholmod_sparse),           # Bset: Sparse subset of B
            POINTER(POINTER(cholmod_dense)),   # X_Handle: Solution handle
            POINTER(POINTER(cholmod_sparse)),  # Xset_Handle: Sparse solution handle
            POINTER(POINTER(cholmod_dense)),   # Y_Handle: Workspace handle
            POINTER(POINTER(cholmod_dense)),   # E_Handle: Workspace handle
            POINTER(cholmod_common),           # Common: Workspace/Parameters
        ]

        # For a general 'b' vector, lots of data
        dll.cholmod_solve.argtypes = [
            c_int,                             # sys: System to solve
            POINTER(cholmod_factor),           # L: Factorization
            POINTER(cholmod_dense),            # B: Right hand side
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

        dll.cholmod_sdmult.argtypes = [
            POINTER(cholmod_sparse),           # A: Sparse matrix
            c_int,                             # transpose: 0=A, 1=A', 2=A.'
            POINTER(c_double),                 # alpha: Scalar (alpha[2])
            POINTER(c_double),                 # beta: Scalar (beta[2])
            POINTER(cholmod_dense),            # X: Dense vector/matrix
            POINTER(cholmod_dense),            # Y: Output dense vector/matrix
            POINTER(cholmod_common),           # Common: Workspace/Parameters
        ]

        # Permutation func needed for updown
        dll.cholmod_submatrix.argtypes = [
            POINTER(cholmod_sparse),           # A: Matrix to slice
            POINTER(c_int32),                  # rset: Row indices
            c_int64,                           # rsize: Size of rset
            POINTER(c_int32),                  # cset: Column indices
            c_int64,                           # csize: Size of cset
            c_int,                             # values: Pattern/Real/Complex
            c_int,                             # sorted: Sort result?
            POINTER(cholmod_common),           # Common: Workspace/Parameters
        ]

        # Update Graph
        dll.cholmod_updown.argtypes = [
            c_int,                             # update: 1=update, 0=downdate
            POINTER(cholmod_sparse),           # C: Rank-k update matrix
            POINTER(cholmod_factor),           # L: Factorization to modify
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

        dll.cholmod_updown_solve.argtypes = [
            c_int,                             # update: 1=update, 0=downdate
            POINTER(cholmod_sparse),           # C: Rank-k update matrix
            POINTER(cholmod_factor),           # L: Factorization
            POINTER(cholmod_dense),            # X: Solution
            POINTER(cholmod_dense),            # DeltaB: Change in RHS
            POINTER(cholmod_common),           # Common: Workspace/Parameters
        ]

        dll.cholmod_allocate_sparse.argtypes = [
            c_size_t,                          # nrow: Number of rows
            c_size_t,                          # ncol: Number of columns
            c_size_t,                          # nzmax: Max non-zeros
            c_int,                             # sorted: Columns sorted?
            c_int,                             # packed: Columns packed?
            c_int,                             # stype: Symmetry type
            c_int,                             # xtype: Pattern/Real/Complex
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

        dll.cholmod_start.argtypes = [POINTER(cholmod_common)]
        dll.cholmod_finish.argtypes = [POINTER(cholmod_common)]

        dll.cholmod_free_factor.argtypes = [
            POINTER(POINTER(cholmod_factor)),  # L: Factor to free
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]
        dll.cholmod_free_dense.argtypes = [
            POINTER(POINTER(cholmod_dense)),   # X: Dense matrix to free
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]
        dll.cholmod_free_sparse.argtypes = [
            POINTER(POINTER(cholmod_sparse)),  # A: Sparse matrix to free
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

        dll.cholmod_allocate_dense.argtypes = [
            c_size_t,                          # nrow: Number of rows
            c_size_t,                          # ncol: Number of columns
            c_size_t,                          # d: Leading dimension
            c_int,                             # xtype: Pattern/Real/Complex
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]
        
        # Allocate an empty factor object
        dll.cholmod_alloc_factor.argtypes = [
            c_size_t,                          # n: Matrix dimension
            c_int,                             # dtype: Double/Single
            POINTER(cholmod_common),           # Common: Workspace/Parameters
        ]
        
        dll.cholmod_zeros.argtypes = [
            c_size_t,                          # nrow: Number of rows
            c_size_t,                          # ncol: Number of columns
            c_int,                             # xtype: Pattern/Real/Complex
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

        # COPY FUNCTIONS
        dll.cholmod_copy_factor.argtypes = [
            POINTER(cholmod_factor),           # L: Factor to copy
            POINTER(cholmod_common),           # Common: Workspace/Parameters
        ]

        # Extras
        # For sparse 'b' vector, like an impulse
        dll.cholmod_spsolve.argtypes = [
            c_int,                             # sys: System to solve
            POINTER(cholmod_factor),           # L: Factorization
            POINTER(cholmod_dense),            # B: Right hand side
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

        dll.cholmod_norm_sparse.argtypes = [
            POINTER(cholmod_sparse),           # A: Sparse matrix
            c_int,                             # norm: 0=inf, 1=1
            POINTER(cholmod_common)            # Common: Workspace/Parameters
        ]

    def config_return_types(self, dll):

        dll.cholmod_analyze.restype = POINTER(cholmod_factor)
        dll.cholmod_factorize_p.restype = c_int
        dll.cholmod_solve2.restype = c_int  
        dll.cholmod_solve.restype = POINTER(cholmod_dense)
        dll.cholmod_sdmult.restype = c_int
        dll.cholmod_submatrix.restype = POINTER(cholmod_sparse)
        dll.cholmod_updown.restype = c_int
        dll.cholmod_updown_solve.restype = c_int
        dll.cholmod_allocate_sparse.restype = POINTER(cholmod_sparse)
        dll.cholmod_start.restype = None
        dll.cholmod_finish.restype = None
        dll.cholmod_free_factor.restype = None
        dll.cholmod_free_dense.restype = None
        dll.cholmod_free_sparse.restype = None
        dll.cholmod_allocate_dense.restype = POINTER(cholmod_dense)
        dll.cholmod_alloc_factor.restype = POINTER(cholmod_factor)
        dll.cholmod_zeros.restype = POINTER(cholmod_dense)
        dll.cholmod_copy_factor.restype = POINTER(cholmod_factor)
        
        # Extras
        dll.cholmod_spsolve.restype = POINTER(cholmod_sparse)
        dll.cholmod_norm_sparse.restype = c_double


    
