"""Uncertainty bound functions.

James Forbes and Steven Dahdah
2022/02/06

To use this module, first call::

    R = unc_bound.residuals(P_nominal, P_off_nominal)

to get the list of residual transfer functions. Then call::

    mag_dB_max, mag_abs_max = unc_bound.residual_max_mag(R, w_shared)

to compute the worst-case residual magnitude over all frequencies. Using that
information, call::

    x_opt, f_opt, f_hist, x_hist = unc_bound.run_optimization(
        x0,
        lb,
        ub,
        max_iter,
        w_shared,
        mag_abs_max,
    )

    W2 = unc_bound.extract_W2(x_opt)

to compute the optimal weight ``W2(s)``. You will need to first set up the
initial design variables ``x0``, along with their upper and lower bounds ``lb``
and ``ub``. You will need to adjust these, along with ``max_iter`` to get a
good fit. The design variables, which parameterize a transfer function, are::

    x = [wa1, za1, wa2, za2, wb1, zb1, wb2, zb2, kappa]

The transfer function they parameterize is::

                    ((s / wb1)**2 + (2 * zb1 / wb1 * s) + 1) * ((s / wb2)**2 + (2 * zb2 / wb2 * s) + 1)  # noqa
    W2(s) = kappa * -----------------------------------------------------------------------------------
                    ((s / wa1)**2 + (2 * za1 / wa1 * s) + 1) * ((s / wa2)**2 + (2 * za2 / wa2 * s) + 1)  # noqa

The function ``extract_W2()`` extracts a transfer function from the design
variables in ``x_opt``.
"""

import control
# %%
import numpy as np
from scipy import optimize as opt


def residuals(P_nom, P):
    """Compute the residuals between P_nom and P.

    Parameters
    ----------
    P_nom : control.TransferFunction
        Nominal transfer function.
    P : List[control.TransferFunction]
        Off-nominal transfer functions.

    Returns
    -------
    List[control.TransferFunction]
        Residual between the nominal transfer function and each off-nominal
        transfer funciton.
    """
    # Number of off-nominal plants.
    R = [plant / P_nom - 1 for plant in P]
    return R


def residual_max_mag(R, w_shared):
    """Compute the max of all the residuals.

    Parameters
    ----------
    R : List[control.TransferFunction]
        Residual transfer functions
    w_shared : np.ndarray
        Frequencies to evaluate TF at (rad/s)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Maximum of all residuals in dB and absolute value respectively.
    """
    # Compute all magnitudes
    mag = np.vstack([
        control.bode(residual, w_shared, plot=False)[0] for residual in R
    ])
    # Compute maximum magnitude at each freq
    mag_abs_max = np.max(mag, axis=0)
    # Compute dB
    mag_dB_max = 20 * np.log10(mag_abs_max)
    return mag_dB_max, mag_abs_max


def extract_W2(x):
    """Extract weighting function from design variables.

    Parameters
    ----------
    x : np.ndarray
        Design variables.

    Returns
    -------
    control.TransferFunction
        ``W2(s)`` transfer function function.
    """
    # Extract DC gain from design variables
    dc_gain = x[-1]
    # Split x up into denomenator and numerator parts (and exlude the DC gain)
    x_split = np.split(x[:-1], 2)
    # Denominator desgin variables
    x_den = x_split[0]
    # Numerator desgin variables
    x_num = x_split[1]

    # Laplace variable s
    s = control.tf('s')
    # Initialize W2 as the DC gain
    W2 = dc_gain
    # Form rest of TF
    for i in range(0, x_den.size, 2):
        wa, za = x_den[i], x_den[i + 1]
        wb, zb = x_num[i], x_num[i + 1]
        W2 = W2 * (((s / wb)**2 + 2 * zb / wb * s + 1)
                   / ((s / wa)**2 + 2 * za / wa * s + 1))
    
    W2_den = np.array(W2.den).ravel()
    W2_num = np.array(W2.num).ravel()
    W2 = control.tf(W2_num / W2_den[0], W2_den / W2_den[0])
    
    # Return W2 transfer function
    return W2


def run_optimization(x0, lb, ub, max_iter, params, constraint_params):
    """Run the weight optimization.

    Design variables are ``[wa1, za1, wa2, za2, wb1, zb1, wb2, zb2, kappa]``.

    Parameters
    ----------
    x0 : np.ndarray
        Initial condition for the optimization design variables
    lb : np.ndarray
        Lower bound constraint on design variables.
    ub : np.ndarray
        Upper bound constraint on design variables.
    params : np.ndarray
        Objective function parameters. Shared frequencies where TF is
        evaluated.
    constraint_params : np.ndarray
        Constraint function parameters. Maximum residual magnitude.

    Returns
    -------
    Tuple[np.ndarray, float, List[float], List[np.ndarray]]
        Optimization results. Specifically, optimal design variables, optimal
        objective value, objective function history, and previous iteration's
        design variables.
    """
    # To keep track of design variable history
    x_hist = []
    # To keep track of objective function history
    f_hist = []
    # To keep track of constraint function history
    g_hist = []

    def objcon(x):
        """Return both the objective function and constraint.

        Also tracks history of objective function through global variable
        ``f_hist``.
        """
        # Call the function structure that will output mass and stress.
        f, g = _obj_and_constraints(x, params, constraint_params)
        return f, g

    def obj(x):
        """Compute objective function to be passed to the optimizer.

        Tracks design variable history in global ``x_hist``, objective function
        history in ``f_hist``, and constraint history in ``g_hist``.

        Only re-computes objective function if design variables change since
        last iteration.
        """
        # Use these functions from the outer scope. The nonlocal needed becuase
        # x_hist, f_hist, and g_hist are being *updated*.
        nonlocal x_hist, f_hist
        # If x and x_hist are not the same, update x, the objective function,
        # and constraint.
        if not np.array_equal(x, x_hist):
            f, g = objcon(x)
            # Append objective history.
            f_hist.append(f)

            # Append design variable history  (not done again in con so we don't add it twice)
            x_hist.append(x)
        return f_hist[-1]

    def con(x):
        """Compute constraint function to be passed to the optimizer.

        Tracks design variable history in global ``x_hist``, objective function
        history in ``f_hist``, and constraint history in ``g_hist``.

        Only re-computes constraint function if design variables change since
        last iteration.
        """
        # Use these functions from the outer scope. The nonlocal needed becuase
        # x_hist, f_hist, and g_hist are being *updated*.
        nonlocal x_hist, g_hist
        # If x and x_hist are not the same, update x, the objective function,
        # and constraint.
        if not np.array_equal(x, x_hist):
            f, g = objcon(x)
            # Append constraint function history.
            g_hist.append(g)
        return g_hist[-1]

    # Set up optimization problem with ``scipy.optimize``

    # Inequality constraint dictionary
    ineq_constraints = {
        'type': 'ineq',
        'fun': con,
    }
    # Specify bounds in the scipy optimize format.
    bounds = opt.Bounds(lb, ub, keep_feasible=True)
    # Run optimization
    res = opt.minimize(
        obj,
        x0,
        method='SLSQP',
        constraints=ineq_constraints,
        bounds=bounds,
        options={
            'ftol': 1e-4,
            'disp': True,
            'maxiter': max_iter,
        },
    )
    # Print results
    print('x =', res.x)
    print('f(x) =', res.fun)
    print('success =', res.success)
    return res.x, res.fun, f_hist, x_hist


def _obj_and_constraints(x, params, constraint_params):
    """Compute objective function and constraints for optimizer.

    Parameters
    ----------
    x : np.ndarray
        Design variables.
    params : np.ndarray
        Objective function parameters. Shared frequencies where TF is
        evaluated.
    constraint_params : np.ndarray
        Constraint function parameters. Maximum residual magnitude.

    Returns
    -------
    Tuple[float, float]
        Objective function and constraint function values for given design
        variables.
    """
    w_shared = params
    mag_abs_max = constraint_params

    # Extract W2 from design variable array.
    W2 = extract_W2(x)

    # Compute freq response of W2.
    mag_abs_W2, phase_rad_W2, w = control.bode(W2, w_shared, plot=False)

    # Compute the error.
    e = mag_abs_W2 - mag_abs_max

    # Compute the objective function.
    f = sum(e**2)

    # Compute the inequality constraitns.
    # constraints in form "constraint >= 0"
    g = e

    return f, g
