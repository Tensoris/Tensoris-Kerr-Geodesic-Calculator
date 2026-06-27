"""
core/metric.py

The Kerr Metric in Boyer-Lindquist coordinates.
Provides the inverse metric components and Hamiltonian
for a rotating (Kerr) black hole spacetime.

Uses Numba JIT compilation for near-C-speed performance.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def kerr_metric_inverse(r, theta, a, M=1.0):
    """
    Returns the inverse Kerr metric components
    in Boyer-Lindquist coordinates (t, r, theta, phi).

    Parameters
    ----------
    r : float
        Radial coordinate
    theta : float
        Polar angle (0 = north pole, pi/2 = equator)
    a : float
        Spin parameter (0 <= a < M)
    M : float
        Black hole mass (default=1.0 for natural units)

    Returns
    -------
    tuple (g_tt, g_rr, g_thth, g_phph, g_tph)
        The non-zero inverse metric components.
        g_tph is the off-diagonal t-phi component.
    """
    Delta = r ** 2 - 2.0 * M * r + a ** 2
    Sigma = r ** 2 + a ** 2 * np.cos(theta) ** 2

    # Precompute common terms
    sin2 = np.sin(theta) ** 2
    Sigma_Delta = Sigma * Delta

    # Inverse metric components
    g_tt = -((r ** 2 + a ** 2) ** 2 - a ** 2 * Delta * sin2) / Sigma_Delta
    g_rr = Delta / Sigma
    g_thth = 1.0 / Sigma
    g_phph = (Delta - a ** 2 * sin2) / (Sigma_Delta * sin2)
    g_tph = -2.0 * M * a * r / Sigma_Delta

    return g_tt, g_rr, g_thth, g_phph, g_tph


@njit(cache=True)
def hamiltonian(r, theta, p_t, p_r, p_theta, p_phi, a, M=1.0):
    """
    Computes the Kerr Hamiltonian.

    H = 1/2 * g^{mu nu} * p_mu * p_nu

    For massive particles: H = -1/2 (conserved)
    For photons:          H = 0    (conserved)

    Parameters
    ----------
    r, theta : float
        Position coordinates
    p_t, p_r, p_theta, p_phi : float
        Conjugate momenta
    a : float
        Black hole spin
    M : float
        Black hole mass

    Returns
    -------
    float
        The Hamiltonian value
    """
    g_tt, g_rr, g_thth, g_phph, g_tph = kerr_metric_inverse(r, theta, a, M)

    H = 0.5 * (
        g_tt * p_t ** 2
        + g_rr * p_r ** 2
        + g_thth * p_theta ** 2
        + g_phph * p_phi ** 2
        + 2.0 * g_tph * p_t * p_phi
    )

    return H


@njit(cache=True)
def hamiltonian_from_state(state, a, M=1.0):
    """
    Convenience wrapper: computes H from a full 8-component state vector.

    state = [t, r, theta, phi, p_t, p_r, p_theta, p_phi]
    """
    _, r, theta, _, _, p_t, p_r, p_theta, p_phi = state
    return hamiltonian(r, theta, p_t, p_r, p_theta, p_phi, a, M)


def compute_carter_constant(r, theta, p_t, p_r, p_theta, p_phi, a, M=1.0):
    """
    Computes the Carter Constant Q, a conserved quantity in Kerr spacetime.

    Q = p_theta^2 + cos^2(theta) * (a^2 * (mu^2 - p_t^2) + p_phi^2 / sin^2(theta))

    where mu^2 = -2*H (so mu=1 for massive particles with H=-1/2).

    The Carter Constant provides the most sophisticated conservation check
    in Kerr spacetime, representing the total "non-equatorial" motion.

    Parameters
    ----------
    r, theta : float
        Position coordinates (r is not used in the standard expression,
        but kept for API consistency)
    p_t, p_r, p_theta, p_phi : float
        Conjugate momenta
    a : float
        Black hole spin
    M : float
        Black hole mass

    Returns
    -------
    float
        The Carter Constant Q
    """
    H = hamiltonian(r, theta, p_t, p_r, p_theta, p_phi, a, M)
    mu_squared = -2.0 * H  # = 1 for massive particles with H = -1/2

    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2

    Q = (
        p_theta ** 2
        + cos2 * (a ** 2 * (mu_squared - p_t ** 2) + p_phi ** 2 / sin2)
    )

    return Q
