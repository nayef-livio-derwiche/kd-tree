# -*- coding: utf-8 -*-
"""
Gram-Schmidt Orthonormalizaion algorithm implemented with cython to be fast.
It is mainly used here to generate randomly rotation matrix.

Note: the distribution is not uniform on the rotation group, but we do not need
it. 
"""
import cython
from libc.math cimport sqrt
import numpy as np
cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef dot(double[:] u, double[:] v):
    cdef int i
    cdef int n = u.shape[0]
    cdef double s = 0
    for i in range(n):
        s += u[i] * v[i]
    return s

cdef np.ndarray proj(np.ndarray v, np.ndarray u):
    return dot(u,v)/dot(u,u)*u
   
@cython.boundscheck(False)
def gram_schmidt(int d):
    """
    Gram-Schmidt implementation slightly faster than vanilla python. 
    """
    cdef int i, j, k
    cdef double c
    cdef np.ndarray[DTYPE_t, ndim=2] O = np.random.rand(d,d)
    cdef double[:] u
    for i in range(1,d):
        u = O[:,i]
        for j in range(i):
            c = np.dot(u,O[:,j])/np.dot(u,u)
            for k in range(d):
                O[k,i] -= c * O[k,j]
            
    return O / np.sqrt(np.square(O).sum(0))
    
@cython.boundscheck(False)
@cython.cdivision(True)
def fast_gram_schmidt(int d):
    """
    Fast Gram-Schmidt implementation using cython.
    """
    cdef int i, j, k
    cdef double c, c1, c2
    cdef np.ndarray[DTYPE_t, ndim=2] O = np.random.rand(d,d)
    for i in range(1,d):
        for j in range(i):
            c1 = 0
            c2 = 0
            for k in range(d):
                c1 += O[k,i] * O[k,j]
                c2 += O[k,j] * O[k,j]
            c = c1 / c2
            for k in range(d):
                O[k,i] -= c * O[k,j]
    for i in range(d):
        c = 0
        for k in range(d):
            c += O[k,i]*O[k,i]
        c = sqrt(c)
        for k in range(d):
            O[k,i] = O[k,i] / c
            
    return O