This data corresponds to the 1D simulation of a wildland fire model.

* "1D_grid.npy = [X_grid  Y_grid]" contains the X and Y grid points.
* "SnapShotMatrix558.49.npy = q" contains the snapshot data for Temperature(T) and Supply mass fraction(S). T =  q[:Nx, :] and S = q[Nx:, :]. 
* "Shifts558.49.npy = s" contains the shifts for the three frames. s[0, :], s[1, :], s[2, :] are the shifts for three frames.
* "Time.npy" contains the time discretization array.
