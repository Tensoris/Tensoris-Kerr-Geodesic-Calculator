"""
utils/constants.py

Physical constants for the Kerr Engine.
All values in SI units unless otherwise noted.
"""

# --- Fundamental Constants ---
G = 6.67430e-11       # Gravitational constant (m^3 kg^-1 s^-2)
c = 2.99792458e8      # Speed of light (m/s)
h = 6.62607015e-34    # Planck constant (J*s)
hbar = 1.054571817e-34  # Reduced Planck constant (J*s)

# --- Solar Mass Reference ---
M_sun = 1.98847e30    # Solar mass (kg)

# --- Black Hole Parameters (Natural Units) ---
# For M=1 in natural units (G=c=1):
# Mass in kg: M_kg = M * (c^2 / G) ... but we use M=1 throughout

# --- Conversion Factors ---
# 1 M (natural) in meters (gravitational radius)
def mass_to_gravradius(M, mass_unit="solar"):
    """
    Convert mass to gravitational radius r_g = GM/c^2.
    
    Parameters
    ----------
    M : float
        Mass value
    mass_unit : str
        'solar' for solar masses, 'kg' for kilograms
    
    Returns
    -------
    float
        Gravitational radius in meters
    """
    if mass_unit == "solar":
        mass_kg = M * M_sun
    elif mass_unit == "kg":
        mass_kg = M
    else:
        raise ValueError(f"Unknown mass_unit: {mass_unit}")
    
    return G * mass_kg / c ** 2


# --- Kerr Metric Parameters ---
# Event horizon: r_plus = M + sqrt(M^2 - a^2)
# Ergosphere:    r_ergo = M + sqrt(M^2 - a^2 * cos^2(theta))
# ISCO:          Innermost Stable Circular Orbit (depends on a and pro/retro)

# --- Simulation Defaults ---
DEFAULT_M = 1.0           # Natural units
DEFAULT_STEPS = 50000     # Integration steps
DEFAULT_LAMBDA_MAX = 2000  # Maximum affine parameter
