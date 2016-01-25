import scipy.optimize
import numpy as np
cimport numpy as np
from math import cos, sin, atan, log

cdef unsigned int L = 3

DTYPE = np.float
ctypedef np.float DTYPE_t

cimport cython
@cython.boundscheck(False)
def f(double x, double a0, double b0, np.ndarray[double, ndim=1] p0, np.ndarray[double, ndim=1] q0, unsigned int i0):
    cdef double s = 0.0
    cdef unsigned int i
    for i in range(i0 + 1):
        s += p0[i] / (x - q0[i])
    return x - a0 - b0 * s


cimport cython
@cython.boundscheck(False)
def find_root(double a, double b, double a0, double b0, np.ndarray[double, ndim=1] p0, np.ndarray[double, ndim=1] q0, unsigned int i0):
    return scipy.optimize.brentq(f, a, b, args=(a0, b0, p0, q0, i0), xtol=1e-12)


cimport cython
@cython.boundscheck(False)
def fr(double a, double b, np.ndarray[double, ndim=2] t, np.ndarray[double, ndim=1] p0, np.ndarray[double, ndim=1] q0, unsigned int i0):
    cdef np.ndarray[double, ndim=1] z = np.zeros((t.shape[1],))
    cdef unsigned int j, i, p
    cdef unsigned int first_infinitesimal = 0

    while first_infinitesimal < 2 * L and abs(t[0][first_infinitesimal] * b) <= 1e-12 * 1e4:
        first_infinitesimal += 1

    if first_infinitesimal == 2 * L:
        t[0][0] = 1
        t[1][0] = a
        return 0

    p0[0] = t[0][first_infinitesimal]
    q0[0] = t[1][first_infinitesimal]

    j = 0
    for i in range(first_infinitesimal + 1, i0 + 1):
        if abs(t[0][i] * b) >= 1e-12 * 1e4:
            if abs(t[1][i] - q0[j]) >= 1e-12 * 1e4:
                j += 1
                p0[j] = t[0][i]
                q0[j] = t[1][i]
            else:
                p0[j] = p0[j] + t[0][i]

    cdef double left_bound = q0[0] - 1e4
    cdef double right_bound = q0[j] + 1e4

    z[0] = find_root(left_bound, q0[0] - 1e-12 * 1e1, a, b, p0, q0, j)

    i = 0
    for i in range(0, j):
        z[i + 1] = find_root(q0[i] + 1e-12 * 1e1, q0[i + 1] - 1e-12 * 1e1, a, b, p0, q0, j)

    z[j + 1] = find_root(q0[j] + 1e-12 * 1e1, right_bound, a, b, p0, q0, j)

    cdef double p_term, p_denom

    i = 0
    for i in range(0, j + 2):
        p_term = 1.0
        p_denom = 1.0

        for p in range(0, j + 1):
            p_term *= z[i] - q0[p]
            if i != p:
                p_denom *= z[i] - z[p]

        if i != j + 1:
            p_denom *= z[i] - z[j + 1]
        t[0][i] = p_term / p_denom
        t[1][i] = z[i]

    return j + 1


cimport cython
@cython.boundscheck(False)
def build_green_fraction(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
    cdef unsigned int imax = len(a) - 1

    cdef np.ndarray[double, ndim=1] p0, q0
    cdef np.ndarray[double, ndim=2] t, up

    p0 = np.zeros((2 * L,), dtype=float)
    q0 = np.zeros((2 * L,), dtype=float)
    t = np.zeros((2, 2 * L), dtype=float)

    t[0][0] = 1
    t[1][0] = a[-1]

    cdef unsigned int i
    cdef unsigned int ii00 = 0
    for i in range(imax - 1, -1, -1):
        ii00 = fr(a[i], b[i], t, p0, q0, ii00)

    up = t.copy()
    np.resize(up, (2, ii00 + 1))
    return up


cimport cython
@cython.boundscheck(False)
def three_diag(np.ndarray[np.complex128_t, ndim=2] H, unsigned int p_index, unsigned int q_index, np.complex128_t k, np.complex128_t m):
    cdef np.ndarray[np.complex128_t, ndim=1] y, y0, y1, a, b, buf
    y0, y1 = np.zeros((2 * L), dtype=complex), np.zeros((2 * L), dtype=complex)
    a, b = np.zeros((2 * L), dtype=complex), np.zeros((2 * L - 1), dtype=complex)

    y0[p_index], y0[q_index] = k, m
    y0 /= np.linalg.norm(y0)
    Hy = np.dot(H, y0)
    a[0] = np.inner(y0.conj(), Hy)
    buf = a[0] * y0

    y = Hy - buf
    cdef double x = np.linalg.norm(y)
    cdef unsigned int imax = 0
    while imax < (2 * L - 1) and x * x > 1e-12 * 1e4:
        b[imax] = x
        y1 = y / b[imax]
        Hy = np.dot(H, y1)
        a[imax + 1] = np.inner(y1.conj(), Hy)
        y = Hy - b[imax] * y0 - a[imax + 1] * y1
        y0 = y1
        x = np.linalg.norm(y)
        imax += 1
    a = np.resize(a, (imax + 1))
    return [z.real for z in a], [z.real * z.real for z in b]


cimport cython
@cython.boundscheck(False)
def calculate_energy(np.ndarray[double, ndim=2] t):
    cdef double pq, s1, s2, s3
    cdef unsigned int i
    s1 = 0.0
    s2 = 0.0
    s3 = 0.9

    for i in range(t.shape[1]):
        pq = t[0][i] * t[1][i]
        s1 = pq
        s2 = pq * atan(t[1][i])
        s3 = t[0][i] * log(t[1][i] * t[1][i] + 1)

    return s1 * 0.5 - (s2 - 0.5 * s3) * 0.318309886183790671


cimport cython
@cython.boundscheck(False)
def calculate_electron_number(np.ndarray[double, ndim=2] t):
    cdef unsigned int i
    cdef double s2 = 0.0 
    for i in range(t.shape[1]):
        s2 += t[0][i] * atan(t[1][i])
    return 0.5 - s2 * 0.318309886183790671


cimport cython
@cython.boundscheck(False)
def build_hamiltonian(np.ndarray[double, ndim=1] theta_angles, np.ndarray[double, ndim=1] phi_angles, np.ndarray[double, ndim=1] N, np.ndarray[double, ndim=1] M, np.ndarray[double, ndim=1] E0, np.ndarray[double, ndim=1] U0, np.ndarray[double, ndim=2] V):
    cdef np.ndarray[np.complex128_t, ndim=2] H = np.zeros((2 * L, 2 * L,), dtype=complex)
    cdef unsigned int i, j
    cdef double dh = 0.0
    for i in range(L):
        H[i][i] = complex(E0[i] - dh + U0[i] * 0.5 * (N[i] - M[i] * cos(theta_angles[i])), 0)
        H[i + L][i + L] = complex(E0[i] + dh + U0[i] * 0.5 * (N[i] + M[i] * cos(theta_angles[i])), 0)

        for j in range(i + 1, L):
            H[i][j] = H[i + L][j + L] = complex(V[i][j], 0)

        x = U0[i] * M[i] * sin(theta_angles[i]) * 0.5
        H[i, i + L] = complex(x * cos(phi_angles[i]), -x * sin(phi_angles[i]))

    # Creating hermitian matrix
    conjugate_matrix = H.T.conj()
    np.fill_diagonal(conjugate_matrix, complex(0, 0))
    return H + conjugate_matrix


cimport cython
@cython.boundscheck(False)
def self_consistent_solution_threaded(unsigned int site_index, theta_angles, phi_angles, N, M, E0, U0, V):
    cdef unsigned int iteration_count, i
    iteration_count = 0
    i = site_index
    print('Looking for solution on site {}'.format(site_index))
    cdef double new_m, new_n, energy
    cdef np.ndarray[np.complex128_t, ndim=2] H
    new_m = M[i]
    new_n = N[i]
    energy = 0.0

    while iteration_count == 0 or ((abs(new_m - M[i]) > 1e-10 or abs(new_n - N[i]) > 1e-10) and iteration_count < 200):
        iteration_count += 1

        N[i], M[i] = new_n, new_m

        H = build_hamiltonian(theta_angles, phi_angles, N, M, E0, U0, V)

        def produce_green_fraction(hamiltonian, indexes, vectors):
            (a, b) = three_diag(hamiltonian, indexes[0], indexes[1], vectors[0], vectors[1])
            return build_green_fraction(np.array(a), np.array(b))

        green_fraction = produce_green_fraction(H, (i, i), (complex(1), complex(1)))
        n_spin_up, eu = calculate_electron_number(green_fraction), calculate_energy(green_fraction)

        green_fraction = produce_green_fraction(H, (i + L, i + L), (complex(1), complex(1)))
        n_spin_down, ed = calculate_electron_number(green_fraction), calculate_energy(green_fraction)

        base_vectors = [(complex(1, 0), complex(1, 0)), (complex(1, 0), complex(-1, 0)),
                        (complex(0, 1), complex(1, 0)), (complex(0, 1), complex(-1, 0))]

        coefficients = [calculate_electron_number(produce_green_fraction(H, (i, i + L), vectors)) for vectors in
                        base_vectors]

        new_n = n_spin_up + n_spin_down
        new_m = (n_spin_up - n_spin_down) * cos(theta_angles[i]) - ((coefficients[0] - coefficients[1]) * cos(
            phi_angles[i]) - (coefficients[2] - coefficients[3]) * sin(phi_angles[i])) * sin(theta_angles[i])

        energy = eu + ed - U0[i] * (new_n ** 2 - new_m ** 2) * 0.25

    if iteration_count == 200:
        raise Exception("Self0consistent solution search ended up with endless loop")
    print('Iterations took {}'.format(iteration_count))
    return iteration_count == 1, site_index, new_n, new_m, energy

