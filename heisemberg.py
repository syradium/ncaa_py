import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi, sqrt
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
import copy
import random
import time
import datetime
import os


N = 13
# coupling intergral
init_J = [0.3] * 3 + [0.1] * 7 + [0.3] * 3
# magnetic moments
M = [2.2] * 3 + [0.6] * 7 + [2.2] * 3

# Initial guess of phi angles of minimas on energy surfaces
initial_guess = [
        np.array([0] * 3 + [pi, 0, pi, 0, pi, 0, pi] + [0.0] * 3),
        np.array([0, 0, 0, 3.06932607e+00, 2.56265103e+00, 2.05576667e+00, 1.54869391e+00, 1.04163163e+00, 5.34769046e-01, 2.81072522e-02, pi, pi, pi])
]

# anysotropy
K = [1] * 3 + [1e-5] * 7 + [1] * 3

J = np.zeros((N, N))
for i in range(1, N):
    J[i - 1][i] = init_J[i]

J[2, 3] = -J[2, 3]
J[9, 10] = -J[9, 10]

J = (J + J.T)


def plot_energies_along_path(arr):
    energies = arr[:,1]
    energies -= min(energies)

    interp = pchip(arr[:,0], energies)
    xx = np.linspace(arr.min(), arr.max(), 200)

    plt.plot(xx, interp(xx))
    plt.plot(arr[:,0], energies, 'bo')
    plt.grid(True)

    save_fig('energies')


def plot_magnetic_moments(angles, name='cluster'):
    ax = plt.axes()
    ax.set_xlim([-1.5, (N + 1) * 1.5])
    ax.set_ylim([-max(M) - 1, max(M) + 1])

    for i in range(N):
        for i in range(N):
            dx = M[i] * sin(angles[i])
            dy = M[i] * cos(angles[i])
            ax.arrow(i * 1.5, 0, dx, dy, head_width=0.05, head_length=0.1, fc='k', ec='k')

    save_fig(name)


def save_fig(name):
    while True:
        unixtime = time.mktime(datetime.datetime.now().timetuple())
        path = '/usr/share/nginx/html/graphs/{}_{}.png'.format(name, unixtime)
        if not os.path.exists(path):
            plt.savefig(path, dpi=200)
            plt.clf()
            return


def energy(angles, m):
    s1, s2 = 0.0, 0.0
    for i in range(1, N):
        s1 -= K[i] * (m[i] * cos(angles[i])) ** 2
        for j in range(1, N):
            if J[i, j]:
                s2 -= 0.5 * J[i, j] * cos(angles[i] - angles[j]) * m[i] * m[j]
    return s1 + s2


def gradient(angles, m):
    def partial_deriviative(i):
        dEdXi = 2 * cos(angles[i]) * sin(angles[i]) * K[i] * m[i] ** 2
        for j in range(N):
            if J[i, j]:
                dEdXi += 0.5 * J[i, j] * sin(angles[i] - angles[j]) * m[i] * m[j]
        return dEdXi

    return np.array([partial_deriviative(i) for i in range(N)])


def find_minima_numpy(angles, m):
    def func(angles):
        return energy(angles, m)

    def jacobian(angles):
        return gradient(angles, m)

    result = minimize(func, angles, tol=1e-12)
    return result.x


def find_minima_antigradient(angles, m):
    angles = np.copy(angles)
    for j in range(1000000):
        G = gradient(angles, m)
        force = LA.norm(G)

        if force < 1e-8:
            return angles

        angles -= 0.5 * G
#        if random.randint(0, j) < 5:
#            plot(angles)

def path_tanget(i, path, energies):
    E1, E2, E3 = energies[i - 1], energies[i], energies[i + 1]
    tau = np.zeros((N,))

    if E3 > E2 and E2 > E1:
            for j in range(N):
                tau[j] = path[i + 1, j] - path[i, j];
    elif E3 < E2 and E2 < E1:
            for j in range(N):
                tau[j] = path[i, j] - path[i - 1, j];
    else:
        dEmax, dEmin = abs(E3 - E2), abs(E1 - E2);

        if dEmax < dEmin:
            dEmax, dEmin = dEmin, dEmax
        if E3 > E1:
            for j in range(N):
                tau[j] = dEmax * ( path[i + 1, j] - path[i, j] ) + dEmin * ( path[i, j] - path[i - 1, j] );
        else:
            for j in range(N):
                tau[j] = dEmin * ( path[i + 1, j] - path[i, j] ) + dEmax * ( path[i, j] - path[i - 1, j] );

    # length = std::inner_product(tau.begin(), tau.end(), tau.begin(), 0.0);
    # for(int i = 0 ; i < 2 * L; ++i)
    # 	tau[i] /= sqrt(length);
    return tau / LA.norm(tau)


def distances_between_images(path):
    distances = np.array([LA.norm(path[i - 1] - path[i]) for i in range(1, len(path))])
    return distances

def mep(number_of_images, initial_state, final_state):
    path = np.zeros(shape=(number_of_images, N), dtype=float)
    velocities = np.zeros(shape=(number_of_images, N), dtype=float)
    energies = np.zeros((number_of_images,))
    gradients = [None] * number_of_images

    energies[0], energies[-1] = energy(initial_state, M), energy(final_state, M)
    gradients[0], gradients[-1] = gradient(initial_state, M), gradient(final_state, M)

    for i in range(N):
        dh = (final_state[i] - initial_state[i]) / (number_of_images - 1)
        for j in range(number_of_images):
            path[j, i] = initial_state[i] + j * dh

    while True:
        for j in range(1, number_of_images - 1):
            gradients[j], energies[j] = gradient(path[j], M), energy(path[j], M)

        max_energy_point = np.argmax(energies)
        result_force = np.zeros((N,))

        for i in range(1, number_of_images - 1):
            tau_vector = path_tanget(i, path, energies)
            spring_force = np.zeros((N,))
            gradient_projection = 0.0
            spring_force_projection = 0.0

            for j in range(1, N):
                spring_force[j] = 1.5 * (path[i + 1, j] + path[i - 1, j] - 2 * path[i, j])
                gradient_projection += gradients[i][j] * tau_vector[j]
                spring_force_projection += spring_force[j] * tau_vector[j]

            perp_V = np.zeros((N,))
            for j in range(1, N):
                if i == max_energy_point:
                    perp_V[j] = -gradients[i][j] + 2 * tau_vector[j] * gradient_projection
                else:
                    perp_V[j] = -(gradients[i][j] - tau_vector[j] * gradient_projection) + spring_force_projection * tau_vector[j]
                result_force[j] += perp_V[j]

            shift = calculate_step_quick(perp_V, i, velocities)
            for j in range(N):
                path[i, j] += shift[j]

        force = LA.norm(result_force)
        if abs(force) < 1e-5:
            print(distances_between_images(path))
            break
    return path


def calculate_step_quick(perp_v, n, velocities):
    s = np.zeros((N,))
    fv, fd, mass, dt = 0.0, 0.0, 20.0, 0.1

    tmp = 0.0
    for i in range(N):
        tmp = perp_v[i] * velocities[n, i]
        if tmp < 0.0:
            velocities[n, i] = 0.0
        else:
            fv += tmp
        fd += perp_v[i] * perp_v[i]

    tmp = 0.0
    for i in range(N):
        velocities[n, i] = perp_v[i] * (fv / fd + dt / mass)
        s[i] = velocities[n, i] * dt
    if LA.norm(s) > 1:
        s /= LA.norm(s)
    return s


minimas = [find_minima_numpy(i, M) for i in initial_guess]

for m in minimas:
    plot_magnetic_moments(m)

path = mep(25, minimas[0], minimas[1])

for i, image in enumerate(path):
    plot_magnetic_moments(image)

plot_energies_along_path(nd.array([(i, energy(image, M)) for i, image in enumerate(path)]))
