import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi, sqrt
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
import copy
import random

N = 13
# coupling intergral
#init_J = [0.1] * 3 + [0.7] * 7 + [0.1] * 3
init_J = [0.3] * 3 + [0.1] * 7 + [0.3] * 3
# magnetic moments
M = [2.2] * 3 + [0.6] * 7 + [2.2] * 3

# Phi angles
dot1 = np.array([0] * 3 + [pi, 0, pi, 0, pi, 0, pi] + [0.0] * 3)
dot2 = np.array([-8.92412738e-04, -2.08229603e-03, -6.04857611e-03, 3.06932607e+00, 2.56265103e+00, 2.05576667e+00, 1.54869391e+00, 1.04163163e+00, 5.34769046e-01, 2.81072522e-02, 3.14764130e+00, 3.14367498e+00, 3.14248508e+00])

# anysotropy
K = [1e-1] * 3 + [1e-5] * 7 + [1e-1] * 3

J = np.zeros((N, N))
for i in range(1, N):
    J[i - 1][i] = init_J[i]

J[2, 3] = -J[2, 3] * 2
J[9, 10] = -J[9, 10] * 2

J = (J + J.T)


def plot(angles):
    ax = plt.axes()
    ax.set_xlim([-1.5, (N + 1) * 1.5])
    ax.set_ylim([-max(M) - 1, max(M) + 1])

    for i in range(N):
        for i in range(N):
            dx = M[i] * sin(angles[i])
            dy = M[i] * cos(angles[i])
            ax.arrow(i * 1.5, 0, dx, dy, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.show()

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

    result = minimize(func, angles, tol=1e-15)
    print(result)
    return result.x


def find_minima_antigradient(angles, m):
    angles = np.copy(angles)
    for j in range(100000):
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



def mep(number_of_images, initial_state, final_state):
    path = np.ndarray(shape=(number_of_images, N), dtype=float)
    velocities = np.ndarray(shape=(number_of_images, N), dtype=float)
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
            gradients[j] = gradient(path[j], M)
            energies[j] = energy(path[j], M)

        max_energy_point = np.argmin(energies)
        result_force = np.zeros((N,))
        for i in range(1, number_of_images - 1):
            tau_vector = path_tanget(i, path, energies)
            spring_force = [0.0] * N
            gradient_projection = 0.0
            spring_force_projection = 0.0

            for j in range(N):
                spring_force[j] =  0.5 * (path[i + 1, j] + path[i - 1, j] - 2 * path[i, j])
                gradient_projection += gradients[i][j] * tau_vector[j]
                spring_force_projection += spring_force[j] * tau_vector[j]

            perp_V = [0] * N
            for j in range(N):
                if i == max_energy_point:
                    perp_V[j] = -gradients[i][j] + 2 * tau_vector[j] * gradient_projection
                else:
                    perp_V[j] = -(gradients[i][j] - tau_vector[j] * gradient_projection) + spring_force_projection * tau_vector[j]
                result_force[j] += perp_V[j]

            shift = calculate_step_quick(perp_V, i, velocities)
            for j in range(N):
                path[i, j] += shift[j]

        print(LA.norm(result_force))

        if abs(LA.norm(result_force)) < 1e-5:
            break
    return path


def calculate_step_quick(perp_v, n, velocities):
    s = np.zeros((N,))
    fv, fd, mass, dt = 0, 0, 1, 0.1

    tmp = 0
    for i in range(N):
        tmp = perp_v[i] * velocities[n, i]
        if tmp < 0:
            velocities[n, i] = 0
        else:
            fv += tmp
        fd += perp_v[i] * perp_v[i]

    tmp = 0
    for i in range(N):
        velocities[n, i] = perp_v[i] * (fv / fd + dt / mass)
        s[i] = velocities[n, i] * dt

    return s / LA.norm(s)


res1, res2 = find_minima_antigradient(dot1, M), find_minima_antigradient(dot2, M)
result = mep(7, res1, res2)
for i in result:
    plot(i)
#print(res)
