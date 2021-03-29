cimport cython
import numpy as np
cimport numpy as np 

@cython.boundscheck(False)
cpdef to_array(list inp):
    cdef np.int16_t[:] arr = np.zeros(len(inp), dtype=np.int16)
    cdef Py_ssize_t idx
    for idx in range(len(inp)):
        arr[idx] = inp[idx]
    return np.asarray(arr)