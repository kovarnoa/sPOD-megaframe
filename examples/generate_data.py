import numpy as np
from numpy import mod, meshgrid, cos, sin, exp, pi

"""
	Functions generating data for 1D sPOD tests used in main_comparison_traveling_wave_1D.py 
	All examples except for three_cross_waves are Nframes=2
	All examples except those with sine waves have ~ 0 interpolation error
"""
def generate_data(Nx, Nt, case, noise_percent=0.125):
    Tmax = 0.5  # total time
    L = 1  # total domain size
    sigma = 0.015  # standard diviation of the puls
    x = np.arange(0, Nx) / Nx * L
    t = np.arange(0, Nt) / Nt * Tmax
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    c = 1
    [X, T] = meshgrid(x, t)
    X = X.T
    T = T.T

    if case == "crossing_waves":
        nmodes = 1
        fun = lambda x, t: exp(-((mod((x - c * t), L) - 0.1) ** 2) / sigma**2) + exp(
            -((mod((x + c * t), L) - 0.9) ** 2) / sigma**2
        )

        # Define your field as a list of fields:
        # For example the first element in the list can be the density of a
        # flow quantity and the second element could be the velocity in 1D
        density = fun(X, T)
        velocity = fun(X, T)
        shifts1 = np.asarray(-c * t)
        #shifts1 += 0.8
        shifts2 = np.asarray(c * t)
        shifts2 -= 0.8         # shifts must always transform the quantities to the same starting point

        Q = density  # , velocity]
        
        nmodes = [1,1]
        shift_list = [shifts1, shifts2]
    if case == "three_cross_waves":
        nmodes = 1
        fun = lambda x, t: exp(-((mod((x - c * t), L) - 0.1) ** 2) / sigma**2) + exp(
            -((mod((x + c * t), L) - 0.9) ** 2) / sigma**2) + exp(
            -((mod(x, L) - 0.5) ** 2) / sigma**2)

        # Define your field as a list of fields:
        # For example the first element in the list can be the density of a
        # flow quantity and the second element could be the velocity in 1D
        density = fun(X, T)
        velocity = fun(X, T)
        shifts1 = np.asarray(-c * t)
        #shifts1 += 0.8
        shifts2 = np.asarray(c * t)
        shifts2 -= 0.8         # shifts must always transform the quantities to the same starting point

        shifts3 = np.zeros_like(shifts1)
        shifts3 -= 0.4

        Q = density  # , velocity]
        
        nmodes = [1,1,1]
        shift_list = [shifts1, shifts2, shifts3]
        
    elif case == "sine_waves":
        delta = 0.0125
        # First frame
        q1 = np.zeros_like(X)
        shifts1 = -0.25 * cos(7 * pi * t)
        for r in np.arange(1, 5):
            x1 = 0.25 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # Second frame
        c2 = dx / dt
        shifts2 = -c2 * t

        q2 = np.zeros_like(X)

        #x2 = 0.2 + 0.1 - shifts2
        #q2 += sin(2 * pi * T / Tmax) * exp(-((X - x2) ** 2) / delta**2)
        r=1
        x2 = 0.2 + 0.1*r - shifts2
        q2 += sin(2 * pi * r * T / Tmax) * exp(-((X - x2) ** 2) / delta**2)

        shifts2 += 0.05 		# shifts must always transform the quantities to the same starting point

        Q = q1 + q2
        nmodes = [4, 1]
        shift_list = [shifts1, shifts2]

    elif case == "sine_waves_noise":
        delta = 0.0125
        # first frame
        q1 = np.zeros_like(X)
        shifts1 = -0.25 * cos(7 * pi * t)
        for r in np.arange(1, 5):
            x1 = 0.25 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # second frame
        c2 = dx / dt
        shifts2 = -c2 * t

        q2 = np.zeros_like(X)

        x2 = 0.2 + 0.1 - shifts2
        q2 += sin(2 * pi * T / Tmax) * exp(-((X - x2) ** 2) / delta**2)

        shifts2 += 0.05  		# shifts must always transform the quantities to the same starting point

        Q = q1 + q2  # + E
        indices = np.random.choice(
            np.arange(Q.size), replace=False, size=int(Q.size * noise_percent)
        )
        Q = Q.flatten()
        Q[indices] = 1
        Q = np.reshape(Q, np.shape(q1))
        nmodes = [4, 1]
        shift_list = [shifts1, shifts2]

    elif case == "multiple_ranks":
        delta = 0.0125
        # first frame
        q1 = np.zeros_like(X)
        c2 = dx / dt
        shifts1 = c2 * t
        for r in np.arange(1, 5):
            x1 = 0.5 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # second frame
        c2 = dx / dt
        shifts2 = -c2 * t
        q2 = np.zeros_like(X)
        for r in np.arange(1, 3):
            x2 = -0.1 +  0.1 * r - shifts2
            q2 = q2 + sin(2 * pi * r * T / Tmax) * exp(-((X - x2) ** 2) / delta**2)
            #q2 = exp(-((X - x2) ** 2) / delta**2)

        shifts2 += 0.6			# shifts must always transform the quantities to the same starting point

        Q = q1 + q2
        nmodes = [4, 2]
        shift_list = [shifts1, shifts2]
    elif case == "non_cross_mult_ranks":
        delta = 0.0125
        # first frame
        q1 = np.zeros_like(X)
        c2 = dx / dt
        shifts1 = 0.5 *c2 * t
        for r in np.arange(1, 5):
            x1 = 0.75 + 0.05 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # second frame
        c2 = dx / dt
        shifts2 = 1.4 * c2 * t
        q2 = np.zeros_like(X)
        for r in np.arange(1, 3):
            x2 = 0.65 +  0.05 * r - shifts2
            q2 = q2 + sin(2 * pi * r * T / Tmax) * exp(-((X - x2) ** 2) / delta**2)
            #q2 = exp(-((X - x2) ** 2) / delta**2)

        shifts2 += 0.1			# shifts must always transform the quantities to the same starting point

        Q = q1 + q2
        nmodes = [4, 2]
        shift_list = [shifts1, shifts2]
    
    print()
    print("Doing case %s, natural data ranks = %s"%(case, str(nmodes)))
    print()

    return Q, shift_list, nmodes, L, dx



# ============================================================================ #

