from functools import partial
from functools import reduce
import json
from math import cos, sin, atan, log
from multiprocessing import Pool
import logging
from collections import defaultdict

from numpy import matlib
import numpy as np
from numpy import complex
from function import build_green_fraction, three_diag, calculate_energy, calculate_electron_number, build_hamiltonian, self_consistent_solution_threaded


eps = 1e-12
delta = 1e-10
L = 3


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
            return build_green_fraction(np.array(a), np_array(b))

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



def build_energy_surface_threaded(system):
    step_number = 99
    theta2_begin = -1.0 * np.pi
    theta2_end = 2.0 * np.pi
    theta3_begin = -1.0 * np.pi
    theta3_end = 2.0 * np.pi

    theta_angles, phi_angles = np.array(system['theta_angle']), np.array(system['phi_angle'])
    N, M, E0, U0, V = np.array(system['N'], dtype=float), np.array(system['M']), np.array(system['E0']), np.array(system['U0']), np.array(system['hopping_matrix'])

    surface = matlib.zeros((step_number, step_number), dtype=float)

    for i, th2 in enumerate(np.linspace(theta2_begin, theta2_end, step_number)):
        theta_angles[0] = 0.0
        theta_angles[1] = th2

        logging.info('Step {} / {}'.format(i, step_number))
        for j, th3 in enumerate(np.linspace(theta3_begin, theta3_end, step_number)):
            theta_angles[2] = th3

            E = np.zeros((L,))
            N = np.array(N.copy())
            M = np.array(M.copy())

            logging.info('Processing angles: {}, {}, {}'.format(theta_angles[0], theta_angles[1], theta_angles[2]))

            while True:
                is_consistent = True
                sconsist = partial(self_consistent_solution_threaded, theta_angles=theta_angles, phi_angles=phi_angles,
                                   N=N, M=M, E0=E0, U0=U0, V=V)
                with Pool(4) as pool:
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
