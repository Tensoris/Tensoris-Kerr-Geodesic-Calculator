"""
main.py
========
Tensoris-Kerr Engine — Execution Script

Simulates geodesic trajectories around a rotating (Kerr) black hole.

Three test cases:
  1. run_stable_circular_orbit()  — Particle in circular orbit at r=10M
  2. run_plunging_orbit()         — Particle plunges into the event horizon
  3. run_penrose_process()        — Energy extraction via the Penrose Process
  4. run_photon_orbit()           — Light-like geodesic (null trajectory)
"""

import numpy as np
import sys
import os

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.integrator import integrate_geodesic
from utils.visualizer import (
    plot_geodesic,
    plot_conservation,
    plot_carter_constant,
    plot_equatorial_cross_section,
    plot_all_diagnostics,
)


def run_stable_circular_orbit(a=0.9, M=1.0):
    """
    Test Case 1: Stable Circular Orbit at r = 10M.

    A massive particle initialized in the equatorial plane
    with momenta appropriate for a stable circular orbit.
    The radial coordinate should remain close to 10M throughout.

    Parameters
    ----------
    a : float
        Black hole spin (default=0.9)
    M : float
        Black hole mass (default=1.0)

    Returns
    -------
    solution : OdeSolution
    """
    print("\n" + "=" * 60)
    print("  TEST CASE 1: STABLE CIRCULAR ORBIT (r = 10M)")
    print("=" * 60)

    r0 = 10.0
    theta0 = np.pi / 2  # Equatorial plane
    phi0 = 0.0
    t0 = 0.0

    # Initial momenta for a circular orbit
    p_t0 = -1.0          # Energy (negative = forward in time)
    p_r0 = 0.0           # No radial motion
    p_theta0 = 0.0       # No polar motion
    p_phi0 = 3.5         # Approximate angular momentum for circular orbit

    ic = [t0, r0, theta0, phi0, p_t0, p_r0, p_theta0, p_phi0]

    print(f"  Parameters: a={a}M, r0={r0}M, theta0={theta0:.2f} rad")
    print(f"  Momenta: p_t={p_t0}, p_r={p_r0}, p_theta={p_theta0}, p_phi={p_phi0}")

    solution = integrate_geodesic(
        ic, a=a, M=M, lambda_max=1000, steps=30000,
        rtol=1e-10, atol=1e-12, verbose=True
    )

    # Check if r stayed near r0
    r_trajectory = solution.y[1]
    r_mean = np.mean(r_trajectory)
    r_std = np.std(r_trajectory)
    print(f"\n  Orbit stability: mean r = {r_mean:.4f}M, std = {r_std:.4f}M")
    if r_std < 0.5:
        print("  ✅ Circular orbit stable (low radial deviation)")
    else:
        print(f"  ⚠ Large radial variation ({r_std:.2f}M) — orbit may not be perfectly circular")

    return solution


def run_plunging_orbit(a=0.9, M=1.0):
    """
    Test Case 2: Plunging Orbit.

    A particle with inward radial momentum falls into the black hole.
    The integration should terminate at the event horizon.

    Parameters
    ----------
    a : float
        Black hole spin (default=0.9)
    M : float
        Black hole mass (default=1.0)

    Returns
    -------
    solution : OdeSolution
    """
    print("\n" + "=" * 60)
    print("  TEST CASE 2: PLUNGING ORBIT")
    print("=" * 60)

    r0 = 8.0
    theta0 = np.pi / 2  # Equatorial plane
    phi0 = 0.0
    t0 = 0.0

    # Start just inside the prograde ISCO with inward momentum
    p_t0 = -1.0
    p_r0 = -0.5          # Negative = falling inward
    p_theta0 = 0.0
    p_phi0 = 2.5         # Moderate angular momentum

    ic = [t0, r0, theta0, phi0, p_t0, p_r0, p_theta0, p_phi0]

    print(f"  Parameters: a={a}M, r0={r0}M (event horizon: {M + np.sqrt(M**2 - a**2):.3f}M)")
    print(f"  Momenta: p_t={p_t0}, p_r={p_r0}, p_theta={p_theta0}, p_phi={p_phi0}")

    solution = integrate_geodesic(
        ic, a=a, M=M, lambda_max=300, steps=30000,
        rtol=1e-10, atol=1e-12, verbose=True
    )

    # Check if we hit the event horizon
    r_plus = M + np.sqrt(M ** 2 - a ** 2)
    final_r = solution.y[1][-1]
    if final_r <= r_plus * 1.05:
        print(f"\n  ✅ Particle crossed event horizon (final r = {final_r:.4f}M)")
    else:
        print(f"\n  ⚠ Particle did not reach horizon (final r = {final_r:.4f}M)")

    return solution


def run_penrose_process(a=0.99, M=1.0):
    """
    Test Case 3: The Penrose Process.

    A particle enters the ergosphere of a rapidly spinning black hole
    and exits with MORE energy than it entered — extracting rotational
    energy from the black hole.

    The ergosphere for a=0.99M extends from r_plus to r_ergo ≈ 2M
    in the equatorial plane. Inside the ergosphere, the metric component
    g_tt becomes positive (frame-dragging dominates).

    For the Penrose process, the particle must:
    1. Enter the ergosphere with negative p_t (positive energy)
    2. Split inside the ergosphere — one fragment has negative energy
       (p_t > 0) and falls in, the other escapes with excess energy

    Here we simulate a simplified version: a particle enters the ergosphere
    with retrograde angular momentum and positive energy.

    Parameters
    ----------
    a : float
        Black hole spin (default=0.99)
    M : float
        Black hole mass (default=1.0)

    Returns
    -------
    solution : OdeSolution
    """
    print("\n" + "=" * 60)
    print("  TEST CASE 3: PENROSE PROCESS (Energy Extraction)")
    print("=" * 60)

    r_plus = M + np.sqrt(M ** 2 - a ** 2)
    r_ergo_eq = M + np.sqrt(M ** 2 - a ** 2)  # equatorial ergosphere = event horizon

    # Place particle just inside the ergosphere
    r0 = r_ergo_eq * 1.1
    theta0 = np.pi / 2
    phi0 = 0.0
    t0 = 0.0

    # In the ergosphere, negative energy (positive p_t) orbits are possible
    # We use retrograde angular momentum to trigger frame-dragging effects
    p_t0 = 0.3            # Positive p_t = negative energy (bound state)
    p_r0 = 0.5            # Outward radial motion
    p_theta0 = 0.0
    p_phi0 = -1.5         # Retrograde orbit (negative = opposite to spin)

    ic = [t0, r0, theta0, phi0, p_t0, p_r0, p_theta0, p_phi0]

    print(f"  Parameters: a={a}M (near-extremal)")
    print(f"  Event horizon: r+ = {r_plus:.4f}M")
    print(f"  Starting at r0={r0:.4f}M (inside ergosphere)")
    print(f"  p_t={p_t0} (positive = negative energy — Penrose signature)")
    print(f"  p_phi={p_phi0} (retrograde)")

    solution = integrate_geodesic(
        ic, a=a, M=M, lambda_max=500, steps=40000,
        rtol=1e-10, atol=1e-12, terminate_at_horizon=False,
        verbose=True
    )

    # Check energy change
    p_t_final = solution.y[4][-1]
    p_t_initial = ic[4]
    delta_E = -(p_t_final - p_t_initial)  # Energy gain (E = -p_t)
    print(f"\n  Initial p_t = {p_t_initial:.4f}  (E = {-p_t_initial:.4f})")
    print(f"  Final   p_t = {p_t_final:.4f}  (E = {-p_t_final:.4f})")
    print(f"  Energy change: ΔE = {delta_E:.4f}")
    if delta_E > 0:
        print("  ✅ PENROSE PROCESS SIGNATURE: Energy extracted from black hole!")
    else:
        print("  ℹ No net energy extraction in this trajectory")

    return solution


def run_photon_orbit(a=0.5, M=1.0):
    """
    Test Case 4: Photon (Null) Orbit near the photon sphere.

    For a=0.5M, the prograde photon orbit is roughly at r ≈ 2M.
    This tests the massless particle limit where H = 0.

    Parameters
    ----------
    a : float
        Black hole spin (default=0.5)
    M : float
        Black hole mass (default=1.0)

    Returns
    -------
    solution : OdeSolution
    """
    print("\n" + "=" * 60)
    print("  TEST CASE 4: PHOTON ORBIT (Null Geodesic)")
    print("=" * 60)

    r0 = 6.0
    theta0 = np.pi / 2
    phi0 = 0.0
    t0 = 0.0

    # For photon, p_t scales with frequency
    p_t0 = -1.0
    p_r0 = 0.0
    p_theta0 = 0.0
    p_phi0 = 3.0

    ic = [t0, r0, theta0, phi0, p_t0, p_r0, p_theta0, p_phi0]

    print(f"  Parameters: a={a}M, r0={r0}M")
    print(f"  Photon orbit (null geodesic)")

    solution = integrate_geodesic(
        ic, a=a, M=M, lambda_max=1000, steps=30000,
        rtol=1e-10, atol=1e-12, verbose=True
    )

    return solution


def run_all(run_photon=False):
    """
    Runs all test cases sequentially.

    Parameters
    ----------
    run_photon : bool
        If True, also run the photon orbit test case
    """
    print("╔" + "═" * 58 + "╗")
    print("║            TENSORIS-KERR ENGINE                    ║")
    print("║     Kerr Geodesic Integrator — v1.0                 ║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("  Simulating geodesics in Kerr spacetime")
    print(f"  Library: SciPy solve_ivp + Numba JIT")
    print()

    # --- Test Case 1: Stable Circular Orbit ---
    sol1 = run_stable_circular_orbit(a=0.9)
    plot_all_diagnostics(sol1, a=0.9)
    print("\n" + "─" * 60 + "\n")

    # --- Test Case 2: Plunging Orbit ---
    sol2 = run_plunging_orbit(a=0.9)
    plot_geodesic(sol2, a=0.9, filename='kerr_plunging.png',
                  title='Plunging Orbit | a = 0.90M')
    plot_equatorial_cross_section(sol2, a=0.9,
                                  filename='plunging_equatorial.png')
    print("\n" + "─" * 60 + "\n")

    # --- Test Case 3: Penrose Process ---
    sol3 = run_penrose_process(a=0.99)
    plot_geodesic(sol3, a=0.99, filename='kerr_penrose.png',
                  title='Penrose Process | a = 0.99M', color='gold')
    plot_conservation(sol3, a=0.99, filename='penrose_conservation.png')
    plot_equatorial_cross_section(sol3, a=0.99,
                                  filename='penrose_equatorial.png')
    print("\n" + "─" * 60 + "\n")

    # --- Test Case 4: Photon (optional) ---
    if run_photon:
        sol4 = run_photon_orbit(a=0.5)
        plot_geodesic(sol4, a=0.5, filename='kerr_photon.png',
                      title='Photon Orbit | a = 0.50M', color='magenta')
        print("\n" + "─" * 60 + "\n")

    print("\n" + "=" * 60)
    print("  ALL SIMULATIONS COMPLETE")
    print("=" * 60)
    print()
    print("  Output files:")
    print("    • kerr_geodesic.png       — Circular orbit 3D")
    print("    • conservation_check.png  — Hamiltonian conservation")
    print("    • carter_check.png        — Carter constant conservation")
    print("    • equatorial_section.png  — Equatorial projection")
    print("    • kerr_plunging.png       — Plunging orbit 3D")
    print("    • plunging_equatorial.png — Plunging equatorial view")
    print("    • kerr_penrose.png        — Penrose process 3D")
    print("    • penrose_conservation.png— Penrose H-conservation")
    print("    • penrose_equatorial.png  — Penrose equatorial view")
    print()
    print("  Next step: run validator.py to validate against EinsteinPy")


# --- Script Entry Point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tensoris-Kerr Engine: Kerr Geodesic Simulator"
    )
    parser.add_argument('--photon', action='store_true',
                        help='Run photon orbit test case')
    parser.add_argument('--circular', action='store_true',
                        help='Run only the circular orbit test')
    parser.add_argument('--plunging', action='store_true',
                        help='Run only the plunging orbit test')
    parser.add_argument('--penrose', action='store_true',
                        help='Run only the Penrose process test')

    args = parser.parse_args()

    # Check for single-test modes
    if args.circular:
        sol = run_stable_circular_orbit(a=0.9)
        plot_all_diagnostics(sol, a=0.9)
    elif args.plunging:
        sol = run_plunging_orbit(a=0.9)
        plot_geodesic(sol, a=0.9, filename='kerr_plunging.png')
        plot_equatorial_cross_section(sol, a=0.9, filename='plunging_equatorial.png')
    elif args.penrose:
        sol = run_penrose_process(a=0.99)
        plot_geodesic(sol, a=0.99, filename='kerr_penrose.png', color='gold')
        plot_conservation(sol, a=0.99, filename='penrose_conservation.png')
        plot_equatorial_cross_section(sol, a=0.99, filename='penrose_equatorial.png')
    else:
        # Run all test cases
        run_all(run_photon=args.photon)