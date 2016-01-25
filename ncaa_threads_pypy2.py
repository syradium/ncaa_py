from functools import partial
from math import cos, sin, atan, log
from multiprocessing import Pool
import logging

from numpy import matlib
import numpy as np

# import scipy.optimize


eps = 1e-12
delta = 1e-10
L = 3

def build_hamiltonian(theta_angles, phi_angles, N, M, E0, U0, V, dh = 0):
    H = np.zeros((2 * L, 2 * L,), dtype=np.complex)
    for i in range(L):
        H[i][i] = np.complex(E0[i] - dh + U0[i] * 0.5 * (N[i] - M[i] * cos(theta_angles[i])), 0)
        H[i + L][i + L] = np.complex(E0[i] + dh + U0[i] * 0.5 * (N[i] + M[i] * cos(theta_angles[i])), 0)

        for j in range(i + 1, L):
            H[i][j] = H[i + L][j + L] = np.complex(V[i][j], 0)

        x = U0[i] * M[i] * sin(theta_angles[i]) * 0.5
        H[i, i + L] = np.complex(x * cos(phi_angles[i]), -x * sin(phi_angles[i]))

    # Creating hermitian matrix
    conjugate_matrix = H.T.conj()
    np.fill_diagonal(conjugate_matrix, np.complex(0, 0))
    return H + conjugate_matrix


def find_root(a, b, a0, b0, p0, q0, i0):
    def f(x, a0, b0, p0, q0, i0):
        s = sum(map(lambda i: p0[i] / (x - q0[i]), range(i0 + 1)))
        return x - a0 - b0 * s

    l = a
    r = b
    while r - l > eps:
        m = (l + r) / 2
        if f(m, a0, b0, p0, q0, i0) * f(r, a0, b0, p0, q0, i0) <= 0:
            l = m
        else:
            r = m
    return l

def FR(a, b, t, p0, q0, i0):
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
         ii00 = FR(a[i], b[i], t, p0, q0, ii00)

    return (t, ii00)


def CN(t, imax):
    s2 = sum(map(lambda i: t[0][i] * atan(t[1][i]), range(0, imax + 1)))
    return 0.5 - s2 * 0.318309886183790671


def calculate_energy(t, imax):
    s1 = map(lambda i: t[0][i] * t[1][i], range(imax + 1))
    s2 = map(lambda i: t[0][i] * t[1][i] * atan(t[1][i]), range(imax + 1))
    s3 = map(lambda i: t[0][i] * log(t[1][i] * t[1][i] + 1), range(imax + 1))

    return sum(s1) * 0.5 - (sum(s2) - 0.5 * sum(s3)) * 0.318309886183790671


def three_diag(H, p_index, q_index, k, m):
    y0 = np.zeros((2 * L), dtype=np.complex)
    y1 = np.zeros((2 * L), dtype=np.complex)
    y = np.zeros((2 * L), dtype=np.complex)
    a = np.zeros((2 * L), dtype=np.complex)
    b = np.zeros((2 * L - 1), dtype=np.complex)

    y0[p_index], y0[q_index] = k, m
    y0 /= np.linalg.norm(y0)
    Hy = np.dot(H, y0)
    a[0] = sum(map(lambda i: y0[i].conj() * Hy[i], range(2 * L)))

    buf = a[0] * y0

    y = Hy - buf
    x = np.linalg.norm(y)
    imax = 0
    while imax < (2 * L - 1) and x * x > eps * 1e4:
        b[imax] = x
        y1 = y / b[imax]
        Hy = np.dot(H, y1)
        a[imax + 1] = sum(map(lambda i: y1[i].conj() * Hy[i], range(2 * L)))
        y = Hy - b[imax] * y0 - a[imax + 1] * y1
        y0 = y1
        x = np.linalg.norm(y)
        imax += 1
    a = np.resize(a, (imax + 1))
    return [z.real for z in a], [z.real * z.real for z in b]


def self_consistent_solution(site_index, theta_angles, phi_angles, N, M, E0, U0, V, E):
    M1 = M.copy()
    N1 = N.copy()

    iteration_count = 0
    i = site_index
    logging.debug('Looking for solution on site {}'.format(site_index))

    while iteration_count == 0 or ((abs(M1[i] - M[i]) > delta or abs(N1[i] - N[i]) > delta) and iteration_count < 200):
        iteration_count += 1

        N[i], M[i] = N1[i], M1[i]

        H = build_hamiltonian(theta_angles, phi_angles, N, M, E0, U0, V)

        (a, b) = three_diag(H, i, i, np.complex(1, 0), np.complex(1, 0))
        green_fraction, imax = build_green_fraction(a, b)
        Nu = CN(green_fraction, imax)
        Eu = calculate_energy(green_fraction, imax)

        (a, b) = three_diag(H, i + L, i + L, np.complex(1, 0), np.complex(1, 0))
        green_fraction, imax = build_green_fraction(a, b)
        Nd = CN(green_fraction, imax)
        Ed = calculate_energy(green_fraction, imax)

        (a, b) = three_diag(H, i, i + L, np.complex(1, 0), np.complex(1, 0))
        (green_fraction, imax) = build_green_fraction(a, b)
        SFp = CN(green_fraction, imax)

        (a, b) = three_diag(H, i, i + L, np.complex(1, 0), np.complex(-1, 0))
        (green_fraction, imax) = build_green_fraction(a, b)
        SFn = CN(green_fraction, imax)

        (a, b) = three_diag(H, i, i + L, np.complex(0, 1), np.complex(1, 0))
        (green_fraction, imax) = build_green_fraction(a, b)
        AFp = CN(green_fraction, imax)

        (a, b) = three_diag(H, i, i + L, np.complex(0, 1), np.complex(-1, 0))
        (green_fraction, imax) = build_green_fraction(a, b)
        AFn = CN(green_fraction, imax)

        N1[i] = Nu + Nd
        M1[i] = (Nu - Nd) * cos(theta_angles[i]) - ((SFp - SFn) * cos(phi_angles[i]) - (AFp - AFn) * sin(phi_angles[i])) * sin(theta_angles[i])

        E[i] = Eu + Ed - U0[i] * (N1[i] * N1[i] - M1[i] * M1[i]) * 0.25

    N[i], M[i] = N1[i], M1[i]

    if iteration_count == 200:
        raise "Infinite selfconsist"
    logging.debug('Iterations took {}'.format(iteration_count))
    return iteration_count == 1


def build_energy_surface(theta_angles, phi_angles, M, N, E0, U0, V):
    step_number = 15
    theta2_begin = -1.0 * np.pi
    theta2_end = 2.0 * np.pi
    theta3_begin = -1.0 * np.pi
    theta3_end = 2.0 * np.pi

    results = []
    for th2 in range(0, step_number):
        theta_angles[0] = 0.0
        theta_angles[1] = theta2_begin + (theta2_end - theta2_begin) * th2 / (step_number - 1)

        for th3 in range(0, step_number):
            theta_angles[2] = theta3_begin + (theta3_end - theta3_begin) * th3 / (step_number - 1)

            E = np.zeros((L,))
            N = N.copy()
            M = M.copy()

            logging.debug('Processing angles: {}, {}, {}'.format(theta_angles[0], theta_angles[1], theta_angles[2]))

            while True:
                is_consistent = True
                for i in range(L):
                    result = self_consistent_solution(i, theta_angles, phi_angles, N, M, E0, U0, V, E)
                    is_consistent &= result
                if is_consistent:
                    break
            logging.debug('Resulting d-electons numbers: {}'.format(N))
            results.append(sum(E))

    return results


def self_consistent_solution_threaded(site_index, theta_angles, phi_angles, N, M, E0, U0, V, E):
    iteration_count = 0
    i = site_index
    logging.debug('Looking for solution on site {}'.format(site_index))
    new_M = M[i]
    new_N = N[i]
    energy = 0.0

    while iteration_count == 0 or ((abs(new_M - M[i]) > delta or abs(new_N - N[i]) > delta) and iteration_count < 200):
        iteration_count += 1

        N[i], M[i] = new_N, new_M

        H = build_hamiltonian(theta_angles, phi_angles, N, M, E0, U0, V)

        (a, b) = three_diag(H, i, i, np.complex(1, 0), np.complex(1, 0))
        green_fraction, imax = build_green_fraction(a, b)
        Nu = CN(green_fraction, imax)
        Eu = calculate_energy(green_fraction, imax)

        (a, b) = three_diag(H, i + L, i + L, np.complex(1, 0), np.complex(1, 0))
        green_fraction, imax = build_green_fraction(a, b)
        Nd = CN(green_fraction, imax)
        Ed = calculate_energy(green_fraction, imax)

        (a, b) = three_diag(H, i, i + L, np.complex(1, 0), np.complex(1, 0))
        (green_fraction, imax) = build_green_fraction(a, b)
        SFp = CN(green_fraction, imax)

        (a, b) = three_diag(H, i, i + L, np.complex(1, 0), np.complex(-1, 0))
        (green_fraction, imax) = build_green_fraction(a, b)
        SFn = CN(green_fraction, imax)

        (a, b) = three_diag(H, i, i + L, np.complex(0, 1), np.complex(1, 0))
        (green_fraction, imax) = build_green_fraction(a, b)
        AFp = CN(green_fraction, imax)

        (a, b) = three_diag(H, i, i + L, np.complex(0, 1), np.complex(-1, 0))
        (green_fraction, imax) = build_green_fraction(a, b)
        AFn = CN(green_fraction, imax)

        new_N = Nu + Nd
        new_M = (Nu - Nd) * cos(theta_angles[i]) - ((SFp - SFn) * cos(phi_angles[i]) - (AFp - AFn) * sin(phi_angles[i])) * sin(theta_angles[i])

        energy = Eu + Ed - U0[i] * (new_N * new_N - new_M * new_M) * 0.25

    if iteration_count == 200:
        raise "Infinite selfconsist"
    logging.debug('Iterations took {}'.format(iteration_count))
    return iteration_count == 1, site_index, new_N, new_M, energy


def build_energy_surface_threaded(theta_angles, phi_angles, M, N, E0, U0, V):
    step_number = 99
    theta2_begin = -1.0 * np.pi
    theta2_end = 2.0 * np.pi
    theta3_begin = -1.0 * np.pi
    theta3_end = 2.0 * np.pi

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
                sconsist = partial(self_consistent_solution_threaded, theta_angles=theta_angles, phi_angles=phi_angles, N=N, M=M, E0=E0, U0=U0, V=V, E=E)
                with Pool(2) as pool:
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

def build_energy_surface_threaded_by_angle(theta_angles, phi_angles, M, N, E0, U0, V):
    step_number = 99
    theta2_begin = -1.0 * np.pi
    theta2_end = 2.0 * np.pi
    theta3_begin = -1.0 * np.pi
    theta3_end = 2.0 * np.pi

    surface = matlib.zeros((step_number, step_number), dtype=float)

    for i, th2 in enumerate(np.linspace(theta2_begin, theta2_end, step_number)):
        theta_angles[0] = 0.0
        theta_angles[1] = th2

        logging.info('Step {} / {}'.format(i, step_number))

        with Pool(1) as pool:
            chunker = partial(build_surface_chunk, theta_angles=theta_angles, phi_angles=phi_angles, N=N, M=M, E0=E0, U0=U0, V=V)
            result = pool.map(chunker, [th3 for th3 in np.linspace(theta3_begin, theta3_end, step_number)])
            for j, energy, _, _ in enumerate(result):
                surface[i, j] = energy


    # Normalize enegries and return result
    return surface - surface.min()

def save_energy_surface(surface, path):
    np.savetxt(path, surface)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


electron_energy = [-12, -12, -12]
coloumb_repulsion = [13, 13, 13]
electron_number = np.array([1.4, 1.4, 1.4])
magnetic_moments = np.array([0.5, 0.5, 0.5])
th_angles = [0.0, 0.0, 0.0]
ph_angles = [0.0, 0.0, 0.0]

hopping_parameters = [
        [0.0, 0.95, 1.2],
        [0.0, 0.0, 1.3],
        [0.0, 0.0, 0.0],
]

results = build_energy_surface(th_angles, ph_angles, magnetic_moments, electron_number, electron_energy, coloumb_repulsion, hopping_parameters)
save_energy_surface(results, 'ASyncFE.txt')
