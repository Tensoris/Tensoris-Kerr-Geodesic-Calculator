"""
utils/visualizer.py

Matplotlib 3D visualization for Kerr geodesic trajectories.
Converts Boyer-Lindquist coordinates to Cartesian for plotting.

Provides:
- plot_geodesic()          : 3D trajectory plot
- plot_conservation()      : Hamiltonian conservation check
- plot_carter_constant()   : Carter constant conservation check
- plot_ergosphere_2d()     : 2D cross-section with ergosphere
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from core.metric import hamiltonian, compute_carter_constant


def _boyer_lindquist_to_cartesian(r, theta, phi):
    """
    Convert Boyer-Lindquist coordinates to Cartesian.

    Parameters
    ----------
    r : array-like
        Radial coordinate
    theta : array-like
        Polar angle
    phi : array-like
        Azimuthal angle

    Returns
    -------
    x, y, z : arrays
        Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def plot_geodesic(solution, a, M=1.0, title=None, save=True, show=True,
                  filename='kerr_geodesic.png', figsize=(14, 12),
                  color='cyan', linewidth=0.8, alpha=0.9):
    """
    Plots the Kerr geodesic in 3D Cartesian coordinates.

    Parameters
    ----------
    solution : OdeSolution
        The integration result from integrate_geodesic()
    a : float
        Black hole spin parameter
    M : float
        Black hole mass
    title : str, optional
        Custom plot title
    save : bool
        Save figure to file
    show : bool
        Display the plot interactively
    filename : str
        Output filename if save=True
    figsize : tuple
        Figure dimensions (width, height)
    color : str
        Geodesic line color
    linewidth : float
        Geodesic line width
    alpha : float
        Geodesic line transparency

    Returns
    -------
    matplotlib.figure.Figure
    """
    r = solution.y[1]
    theta = solution.y[2]
    phi = solution.y[3]

    # Convert to Cartesian
    x, y, z = _boyer_lindquist_to_cartesian(r, theta, phi)

    # Event horizon radius
    r_plus = M + np.sqrt(M ** 2 - a ** 2)

    # Ergosphere radius (equatorial)
    r_ergo = M + np.sqrt(M ** 2 - a ** 2 * np.cos(np.pi / 2) ** 2)

    # Build plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Geodesic trajectory
    ax.plot(x, y, z, color=color, linewidth=linewidth,
            label='Geodesic', alpha=alpha)

    # Mark the initial point
    ax.scatter(x[0], y[0], z[0], color='lime', s=50,
               marker='o', label='Start', zorder=5)

    # Mark the final point
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50,
               marker='x', label='End', zorder=5)

    # Event horizon sphere (wireframe + surface)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_eh = r_plus * np.outer(np.cos(u), np.sin(v))
    y_eh = r_plus * np.outer(np.sin(u), np.sin(v))
    z_eh = r_plus * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_eh, y_eh, z_eh, color='black', alpha=0.5,
                    label='Event Horizon')

    # Ergosphere (outer surface) - equatorial ring highlight
    if a > 0:
        u_ergo = np.linspace(0, 2 * np.pi, 100)
        v_ergo = np.linspace(0, np.pi, 50)
        r_ergo_surf = M + np.sqrt(M**2 - a**2 * np.cos(v_ergo)**2)
        x_ergo = np.outer(np.cos(u_ergo), r_ergo_surf * np.sin(v_ergo))
        y_ergo = np.outer(np.sin(u_ergo), r_ergo_surf * np.sin(v_ergo))
        z_ergo = np.outer(np.ones(np.size(u_ergo)), r_ergo_surf * np.cos(v_ergo))
        ax.plot_wireframe(x_ergo, y_ergo, z_ergo, color='orange',
                          alpha=0.15, rstride=4, cstride=4, label='Ergosphere')

    # Axis labels and styling
    ax.set_xlabel('x / M', color='white')
    ax.set_ylabel('y / M', color='white')
    ax.set_zlabel('z / M', color='white')

    if title is None:
        ax.set_title(f'Kerr Geodesic | Spin: a = {a:.2f}M',
                     color='white', fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')

    # Legend with dark styling
    legend = ax.legend(facecolor='#1a1a1a', labelcolor='white',
                       edgecolor='gray', fontsize=10)
    legend.get_frame().set_alpha(0.8)

    # Dark theme
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')

    # Set equal aspect ratio
    max_range = max(np.max(x), np.max(y), np.max(z), abs(np.min(x)),
                    abs(np.min(y)), abs(np.min(z)))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                    facecolor='#0a0a0a')
        print(f"  Plot saved to: {filename}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_conservation(solution, a, M=1.0, save=True, show=True,
                      filename='conservation_check.png'):
    """
    Plots the Hamiltonian conservation over the affine parameter.
    For a massive particle, H should remain at -0.5 throughout.

    Parameters
    ----------
    solution : OdeSolution
        The integration result
    a, M : float
        Black hole parameters
    save, show : bool
        Save/show flags
    filename : str
        Output filename

    Returns
    -------
    matplotlib.figure.Figure
    """
    lam = solution.t
    H_values = []

    for i in range(len(lam)):
        r = solution.y[1][i]
        theta = solution.y[2][i]
        p_t = solution.y[4][i]
        p_r = solution.y[5][i]
        p_theta = solution.y[6][i]
        p_phi = solution.y[7][i]
        H = hamiltonian(r, theta, p_t, p_r, p_theta, p_phi, a, M)
        H_values.append(H)

    H_values = np.array(H_values)
    H_expected = -0.5 * np.ones_like(lam)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top: H(lambda) trace
    ax1.plot(lam, H_values, 'cyan', linewidth=1, label='H(λ)')
    ax1.axhline(y=-0.5, color='red', linestyle='--', linewidth=1.5,
                label='Expected H = -0.5')
    ax1.fill_between(lam, H_values, -0.5, alpha=0.1, color='cyan')
    ax1.set_xlabel('Affine Parameter λ', color='white')
    ax1.set_ylabel('Hamiltonian H', color='white')
    ax1.set_title('Conservation Check: H(λ) vs λ', color='white',
                  fontsize=13, fontweight='bold')
    ax1.legend(facecolor='#1a1a1a', labelcolor='white')
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#0a0a0a')

    # Bottom: Absolute error from expected
    error = np.abs(H_values - (-0.5))
    ax2.semilogy(lam, error, 'magenta', linewidth=1, label='|H - (-0.5)|')
    ax2.set_xlabel('Affine Parameter λ', color='white')
    ax2.set_ylabel('Absolute Error', color='white')
    ax2.set_title('Hamiltonian Conservation Error', color='white',
                  fontsize=13, fontweight='bold')
    ax2.legend(facecolor='#1a1a1a', labelcolor='white')
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#0a0a0a')

    # Stats text
    max_err = np.max(error)
    mean_err = np.mean(error)
    ax2.text(0.02, 0.95, f'Max Error: {max_err:.2e}\nMean Error: {mean_err:.2e}',
             transform=ax2.transAxes, color='white', fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))

    fig.patch.set_facecolor('#0a0a0a')
    for ax in [ax1, ax2]:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    plt.tight_layout()

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                    facecolor='#0a0a0a')
        print(f"  Conservation plot saved to: {filename}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_carter_constant(solution, a, M=1.0, save=True, show=True,
                         filename='carter_check.png'):
    """
    Plots the Carter Constant Q over the affine parameter.
    Q should be conserved throughout the trajectory.

    Parameters
    ----------
    solution : OdeSolution
        The integration result
    a, M : float
        Black hole parameters
    save, show : bool
        Save/show flags
    filename : str
        Output filename

    Returns
    -------
    matplotlib.figure.Figure
    """
    lam = solution.t
    Q_values = []

    for i in range(len(lam)):
        r = solution.y[1][i]
        theta = solution.y[2][i]
        p_t = solution.y[4][i]
        p_r = solution.y[5][i]
        p_theta = solution.y[6][i]
        p_phi = solution.y[7][i]
        Q = compute_carter_constant(r, theta, p_t, p_r, p_theta, p_phi, a, M)
        Q_values.append(Q)

    Q_values = np.array(Q_values)
    Q_mean = np.mean(Q_values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top: Q(lambda)
    ax1.plot(lam, Q_values, 'gold', linewidth=1, label='Q(λ)')
    ax1.axhline(y=Q_mean, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean Q = {Q_mean:.6f}')
    ax1.set_xlabel('Affine Parameter λ', color='white')
    ax1.set_ylabel('Carter Constant Q', color='white')
    ax1.set_title('Carter Constant Conservation: Q(λ) vs λ', color='white',
                  fontsize=13, fontweight='bold')
    ax1.legend(facecolor='#1a1a1a', labelcolor='white')
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#0a0a0a')

    # Bottom: Relative error
    if abs(Q_mean) > 1e-15:
        rel_error = np.abs((Q_values - Q_mean) / Q_mean)
        ax2.semilogy(lam, rel_error, 'magenta', linewidth=1,
                     label='|Q - Q_mean| / |Q_mean|')
    else:
        rel_error = np.abs(Q_values - Q_mean)
        ax2.semilogy(lam, rel_error, 'magenta', linewidth=1,
                     label='|Q - Q_mean|')
    ax2.set_xlabel('Affine Parameter λ', color='white')
    ax2.set_ylabel('Relative Error', color='white')
    ax2.set_title('Carter Constant Conservation Error', color='white',
                  fontsize=13, fontweight='bold')
    ax2.legend(facecolor='#1a1a1a', labelcolor='white')
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#0a0a0a')

    # Stats
    max_err = np.max(rel_error)
    mean_err = np.mean(rel_error)
    ax2.text(0.02, 0.95, f'Max Rel. Error: {max_err:.2e}\nMean Rel. Error: {mean_err:.2e}',
             transform=ax2.transAxes, color='white', fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))

    fig.patch.set_facecolor('#0a0a0a')
    for ax in [ax1, ax2]:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    plt.tight_layout()

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                    facecolor='#0a0a0a')
        print(f"  Carter constant plot saved to: {filename}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_equatorial_cross_section(solution, a, M=1.0, save=True, show=True,
                                  filename='equatorial_section.png'):
    """
    Plots the geodesic projected onto the equatorial plane (x-y).
    Shows event horizon, ergosphere, and turning points.

    Parameters
    ----------
    solution : OdeSolution
        The integration result
    a, M : float
        Black hole parameters
    save, show : bool
        Save/show flags
    filename : str
        Output filename

    Returns
    -------
    matplotlib.figure.Figure
    """
    r = solution.y[1]
    theta = solution.y[2]
    phi = solution.y[3]
    x, y, z = _boyer_lindquist_to_cartesian(r, theta, phi)

    r_plus = M + np.sqrt(M ** 2 - a ** 2)
    r_ergo_eq = M + np.sqrt(M ** 2 - a ** 2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Geodesic
    ax.plot(x, y, 'cyan', linewidth=1, label='Geodesic', alpha=0.8)
    ax.scatter(x[0], y[0], color='lime', s=50, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], color='red', s=50, marker='x', label='End')

    # Event horizon circle
    horizon_circle = Circle((0, 0), r_plus, color='black', alpha=0.8,
                            label=f'Event Horizon (r={r_plus:.2f}M)')
    ax.add_patch(horizon_circle)

    # Ergosphere circle
    ergo_circle = Circle((0, 0), r_ergo_eq, color='orange', alpha=0.15,
                         label=f'Ergosphere (r={r_ergo_eq:.2f}M)')
    ax.add_patch(ergo_circle)

    # Spin direction arrow
    ax.annotate('', xy=(r_ergo_eq * 0.3, 0), xytext=(-r_ergo_eq * 0.3, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2),
                label='Spin axis')

    ax.set_xlabel('x / M', color='white')
    ax.set_ylabel('y / M', color='white')
    ax.set_title(f'Equatorial Projection | a = {a:.2f}M', color='white',
                 fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='gray',
              loc='upper right')
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    lim = r_ergo_eq * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.15)

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                    facecolor='#0a0a0a')
        print(f"  Equatorial section saved to: {filename}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_all_diagnostics(solution, a, M=1.0):
    """
    Generate all diagnostic plots for a given solution.

    Parameters
    ----------
    solution : OdeSolution
        The integration result
    a, M : float
        Black hole parameters
    """
    print("\n--- Generating Diagnostic Plots ---")
    plot_geodesic(solution, a, M)
    plot_conservation(solution, a, M)
    plot_carter_constant(solution, a, M)
    plot_equatorial_cross_section(solution, a, M)
    print("--- All plots generated ---\n")