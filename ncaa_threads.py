from collections import defaultdict
from constants import *
from functools import partial
from functools import reduce
from math import cos, sin, atan, log
from multiprocessing import Pool
from numpy import complex
from numpy import matlib
import json
import logging
import numpy as np
import scipy.optimize


def build_hamiltonian(theta_angles, phi_angles, N, M, E0, U0, V, dh=0):
    H = np.zeros((2 * L, 2 * L,), dtype=complex)
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


# @jit(float64(float64, float64, float64, float64[:], float64[:], uint32), nopython=True)
def f(x, a0, b0, p0, q0, i0):
    s = 0
    for i in range(i0 + 1):
        s += p0[i] / (x - q0[i])
    return x - a0 - b0 * s


def find_root(a, b, a0, b0, p0, q0, i0):
    return scipy.optimize.brentq(f, a, b, args=(a0, b0, p0, q0, i0), xtol=eps)


def fr(a, b, t, p0, q0, i0):
    z = np.zeros((t.shape[1],), dtype=float)

    first_infinitesimal = 0

    while first_infinitesimal < 2 * L and abs(t[0][first_infinitesimal] * b) <= eps * 1e4:
        first_infinitesimal += 1

    if first_infinitesimal == 2 * L:
        t[0][0] = 1
        t[1][0] = a
        return 0

    p0[0] = t[0][first_infinitesimal]
    q0[0] = t[1][first_infinitesimal]

    j = 0
    for i in range(first_infinitesimal + 1, i0 + 1):
        if abs(t[0][i] * b) >= eps * 1e4:
            if abs(t[1][i] - q0[j]) >= eps * 1e4:
                j += 1
                p0[j] = t[0][i]
                q0[j] = t[1][i]
            else:
                p0[j] += t[0][i]

    left_bound = q0[0] - 1e4
    right_bound = q0[j] + 1e4

    z[0] = find_root(left_bound, q0[0] - eps * 1e1, a, b, p0, q0, j)

    for i in range(0, j):
        z[i + 1] = find_root(q0[i] + eps * 1e1, q0[i + 1] - eps * 1e1, a, b, p0, q0, j)

    z[j + 1] = find_root(q0[j] + eps * 1e1, right_bound, a, b, p0, q0, j)

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


def build_green_fraction(a, b):
    imax = len(a) - 1

    p0 = np.zeros((2 * L,), dtype=float)
    q0 = np.zeros((2 * L,), dtype=float)
    t = np.zeros((2, 2 * L), dtype=float)

    t[0][0] = 1
    t[1][0] = a[-1]

    ii00 = 0
    for i in range(imax - 1, -1, -1):
        ii00 = fr(a[i], b[i], t, p0, q0, ii00)

    up = t.copy()
    np.resize(up, (2, ii00 + 1))
    return up


def calculate_electron_number(t):
    s2 = sum(map(lambda i: t[0][i] * atan(t[1][i]), range(t.shape[1])))
    return 0.5 - s2 * 0.318309886183790671


def calculate_energy(t):
    """
   s1 = 0.0
   s2 = 0.0
   s3 = 0.0

   for i in range(0, imax + 1):
       PQ = t[0][i] * t[1][i]
       s1 += PQ
       s2 += PQ * atan(t[1][i])
       s3 += t[0][i] * log(t[1][i] * t[1][i] + 1)

   return 0.5 * s1 - (s2 - 0.5 * s3) * 0.318309886183790671

   s1 = map(lambda i: t[0][i] * t[1][i], range(imax + 1))
   s2 = map(lambda i: t[0][i] * t[1][i] * atan(t[1][i]), range(imax + 1))
   s3 = map(lambda i: t[0][i] * log(t[1][i] * t[1][i] + 1), range(imax + 1))

   return sum(s1) * 0.5 - (sum(s2) - 0.5 * sum(s3)) * 0.318309886183790671
    """

    def get_sum_term(i):
        pq = t[0][i] * t[1][i]
        s1 = pq
        s2 = pq * atan(t[1][i])
        s3 = t[0][i] * log(t[1][i] * t[1][i] + 1)
        return s1, s2, s3

    result = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]), map(get_sum_term, range(t.shape[1])))
    return result[0] * 0.5 - (result[1] - 0.5 * result[2]) * 0.318309886183790671


def three_diag(H, p_index, q_index, k, m):
    y0, y1 = np.zeros((2 * L), dtype=complex), np.zeros((2 * L), dtype=complex)
    a, b = np.zeros((2 * L), dtype=complex), np.zeros((2 * L - 1), dtype=complex)

    y0[p_index], y0[q_index] = k, m
    y0 /= np.linalg.norm(y0)
    Hy = np.dot(H, y0)
    a[0] = np.inner(y0.conj(), Hy)
    buf = a[0] * y0

    y = Hy - buf
    x = np.linalg.norm(y)
    imax = 0
    while imax < (2 * L - 1) and x * x > eps * 1e4:
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


def self_consistent_solution(site_index, theta_angles, phi_angles, n, m, e0, u0, v, e):
    iteration_count = 0
    i = site_index
    logging.debug('Looking for solution on site {}'.format(site_index))
    new_n, new_m = n[i], m[i]
    while iteration_count == 0 or ((abs(new_m - m[i]) > delta or abs(new_n - n[i]) > delta) and iteration_count < 200):
        iteration_count += 1

        n[i], m[i] = new_n, new_m

        H = build_hamiltonian(theta_angles, phi_angles, n, m, e0, u0, v)

        def green_matrix_element(hamiltonian, indexes, base_vectors):
            (a, b) = three_diag(hamiltonian, indexes[0], indexes[1], base_vectors[0], base_vectors[1])
            return build_green_fraction(a, b)

        green_fraction = green_matrix_element(H, (i, i), (complex(1), complex(1)))
        nu, eu = calculate_electron_number(green_fraction), calculate_energy(green_fraction)

        green_fraction = green_matrix_element(H, (i + L, i + L), (complex(1), complex(1)))
        nd, ed = calculate_electron_number(green_fraction), calculate_energy(green_fraction)

        green_fraction = green_matrix_element(H, (i, i + L), (complex(1), complex(1)))
        sfp = calculate_electron_number(green_fraction)

        green_fraction = green_matrix_element(H, (i, i + L), (complex(1), complex(-1)))
        sfn = calculate_electron_number(green_fraction)

        green_fraction = green_matrix_element(H, (i, i + L), (complex(0, 1), complex(1)))
        afp = calculate_electron_number(green_fraction)

        green_fraction = green_matrix_element(H, (i, i + L), (complex(0, 1), complex(-1)))
        afn = calculate_electron_number(green_fraction)

        new_n = nu + nd
        new_m = (nu - nd) * cos(theta_angles[i]) - ((sfp - sfn) * cos(phi_angles[i]) - (afp - afn) * sin(
            phi_angles[i])) * sin(theta_angles[i])

        e[i] = eu + ed - u0[i] * (new_n ** 2 - new_m ** 2) * 0.25

    n[i], m[i] = new_n, new_m

    if iteration_count == 200:
        raise Exception("Infinite selfconsist")
    logging.debug('Iterations took {}'.format(iteration_count))
    return iteration_count == 1


def build_energy_surface(system, step_number=15):
    theta2_begin, theta2_end = -1.0 * np.pi, 2.0 * np.pi
    theta3_begin, theta3_end = -1.0 * np.pi, 2.0 * np.pi

    theta_angles, phi_angles = system['theta_angle'], system['phi_angle']
    n, m, e0, u0, v = system['N'], system['M'], system['E0'], system['u0'], system['hopping_matrix']
    surface = []
    for th2 in range(0, step_number):
        theta_angles[0] = 0.0
        theta_angles[1] = theta2_begin + (theta2_end - theta2_begin) * th2 / (step_number - 1)

        for th3 in range(0, step_number):
            theta_angles[2] = theta3_begin + (theta3_end - theta3_begin) * th3 / (step_number - 1)

            e = np.zeros((L,))
            n = n.copy()
            m = m.copy()

            logging.debug('Processing angles: {}, {}, {}'.format(theta_angles[0], theta_angles[1], theta_angles[2]))

            is_consistent = False
            while not is_consistent:
                is_consistent = True
                for i in range(L):
                    result = self_consistent_solution(i, theta_angles, phi_angles, n, m, e0, u0, v, e)
                    is_consistent &= result
            logging.debug('Resulting d-electons numbers: {}'.format(n))
            surface.append(sum(e))

    return surface


def self_consistent_solution_threaded(site_index, theta_angles, phi_angles, N, M, E0, U0, V):
    iteration_count = 0
    i = site_index
    logging.debug('Looking for solution on site {}'.format(site_index))
    new_m = M[i]
    new_n = N[i]
    energy = 0.0

    while iteration_count == 0 or ((abs(new_m - M[i]) > delta or abs(new_n - N[i]) > delta) and iteration_count < 200):
        iteration_count += 1

        N[i], M[i] = new_n, new_m

        H = build_hamiltonian(theta_angles, phi_angles, N, M, E0, U0, V)

        def produce_green_fraction(hamiltonian, indexes, vectors):
            (a, b) = three_diag(hamiltonian, indexes[0], indexes[1], vectors[0], vectors[1])
            return build_green_fraction(a, b)

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
    logging.debug('Iterations took {}'.format(iteration_count))
    return iteration_count == 1, site_index, new_n, new_m, energy


def build_energy_surface_threaded(system):
    step_number = 99
    theta2_begin = -1.0 * np.pi
    theta2_end = 2.0 * np.pi
    theta3_begin = -1.0 * np.pi
    theta3_end = 2.0 * np.pi

    theta_angles, phi_angles = system['theta_angle'], system['phi_angle']
    N, M, E0, U0, V = system['N'], system['M'], system['E0'], system['U0'], system['hopping_matrix']

    surface = matlib.zeros((step_number, step_number), dtype=float)

    for i, th2 in enumerate(np.linspace(theta2_begin, theta2_end, step_number)):
        theta_angles[0] = 0.0
        theta_angles[1] = th2

        logging.info('Step {} / {}'.format(i, step_number))
        for j, th3 in enumerate(np.linspace(theta3_begin, theta3_end, step_number)):
            theta_angles[2] = th3

            E = np.zeros((L,))
            N = N.copy()
            M = M.copy()

            logging.info('Processing angles: {}, {}, {}'.format(theta_angles[0], theta_angles[1], theta_angles[2]))

            while True:
                is_consistent = True
                sconsist = partial(self_consistent_solution_threaded, theta_angles=theta_angles, phi_angles=phi_angles,
                                   N=N, M=M, E0=E0, U0=U0, V=V)
                with Pool(1) as pool:
                    result = pool.map(sconsist, range(L))
                    for is_cons, site_index, N1, M1, energy in result:
                        N[site_index], M[site_index], E[site_index] = N1, M1, energy
                        is_consistent = is_consistent & is_cons
                    if is_consistent:
                        break
            logging.debug('Resulting d-electons numbers: {}'.format(N))
            surface[i, j] = sum(E)

    # Normalize enegries and return result
    return surface - surface.min()


def build_surface_chunk(th3, theta_angles, phi_angles, N, M, E0, U0, V):
    theta_angles[2] = th3

    E = np.zeros((L,))
    logging.info('Processing angles: {}, {}, {}'.format(theta_angles[0], theta_angles[1], theta_angles[2]))

    while True:
        is_consistent = True
        for i in range(L):
            result = self_consistent_solution(i, theta_angles, phi_angles, N, M, E0, U0, V, E)
            is_consistent &= result
        if is_consistent:
            break

    logging.debug('Resulting d-electons numbers: {}'.format(N))
    return sum(E), N, M


def build_energy_surface_threaded_by_angle(system):
    step_number = 99
    theta2_begin = -1.0 * np.pi
    theta2_end = 2.0 * np.pi
    theta3_begin = -1.0 * np.pi
    theta3_end = 2.0 * np.pi

    theta_angles, phi_angles = system['theta_angle'], system['phi_angle']
    N, M, E0, U0, V = system['N'], system['M'], system['E0'], system['U0'], system['hopping_matrix']

    surface = matlib.zeros((step_number, step_number), dtype=float)

    for i, th2 in enumerate(np.linspace(theta2_begin, theta2_end, step_number)):
        theta_angles[0] = 0.0
        theta_angles[1] = th2

        logging.info('Step {} / {}'.format(i, step_number))

        with Pool(2) as pool:
            chunker = partial(build_surface_chunk, theta_angles=theta_angles, phi_angles=phi_angles, N=N, M=M, E0=E0,
                              U0=U0, V=V)
            result = pool.map(chunker, [th3 for th3 in np.linspace(theta3_begin, theta3_end, step_number)])
            for j, energy, _, _ in enumerate(result):
                surface[i, j] = energy
    # Normalize energies and return result
    return surface - surface.min()


def save_energy_surface(surface, path):
    np.savetxt(path, surface)


def init():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def load_system_parameters(file_name="system.json"):
    with open(file_name) as file:
        params = json.load(file)
        result = defaultdict(list)
        for site in params['sites']:
            for param in ('E0', 'U0', 'M', 'N', 'phi_angle', 'theta_angle'):
                result[param].append(site[param])
        result['hopping_matrix'] = params['hopping_matrix']
        return result


def main():
    init()
    system = load_system_parameters()
    results = build_energy_surface_threaded(system)
    save_energy_surface(results, 'ASyncFE.txt')


if __name__ == '__main__':
    main()
