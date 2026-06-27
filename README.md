# Tensoris-Kerr Engine

**Kerr Geodesic Integrator** — A professional Python package that simulates the path of light and matter around a rotating black hole using General Relativity.

Built as a "proof of competence" for the convergence meeting with **Prof. Galbiati, Dr. Ianni, and Dr. Peña-Garay.**

---

## Physics Foundation

In General Relativity, we use the **Metric Tensor ($g_{\mu\nu}$)** to describe how spacetime curves around a massive, rotating object. For a rotating black hole, we use the **Kerr Metric** in Boyer-Lindquist coordinates $(t, r, \theta, \phi)$.

### Key Variables
- $M$: Mass (set to 1 in natural units)
- $a$: Spin parameter ($0 \le a < M$)
- $\Delta = r^2 - 2Mr + a^2$
- $\Sigma = r^2 + a^2\cos^2\theta$

### Hamiltonian
For a massive particle: $H = \frac{1}{2}g^{\mu\nu}p_\mu p_\nu = -\frac{1}{2}$

### Conserved Quantities
- **$p_t$**: Energy (time translation symmetry)
- **$p_\phi$**: Angular Momentum (axial symmetry)
- **Carter Constant ($Q$)**: The fourth integral in Kerr spacetime

---

## Project Architecture

```
KerrEngine/
├── core/
│   ├── __init__.py          # Module init
│   ├── metric.py            # Kerr tensor math (Numba-JIT)
│   └── integrator.py        # ODE solver (SciPy solve_ivp)
├── utils/
│   ├── __init__.py          # Module init
│   ├── constants.py         # Physical constants
│   └── visualizer.py        # Matplotlib 3D visualization
├── main.py                  # Execution script (3 test cases)
├── validator.py             # Validation suite + EinsteinPy comparison
└── README.md                # This file
```

---

## Installation

```bash
# Clone or copy the project
cd KerrEngine

# Install dependencies
pip install numpy scipy matplotlib numba einsteinpy sympy

# Or use requirements.txt:
pip install -r requirements.txt
```

---

## Usage

### Run All Test Cases
```bash
python main.py
```

This runs three simulations:
1. **Stable Circular Orbit** at r=10M (a=0.9)
2. **Plunging Orbit** — particle falls into the black hole
3. **Penrose Process** — energy extraction near a rapidly spinning black hole (a=0.99)

### Run Individual Tests
```bash
python main.py --circular     # Circular orbit only
python main.py --plunging     # Plunging orbit only
python main.py --penrose      # Penrose process only
python main.py --photon       # Include photon (null) geodesic
```

### Run Validation Suite
```bash
python validator.py           # Full validation
python validator.py --quick   # Skip EinsteinPy (invariants only)
python validator.py --a 0.99  # Test with different spin
```

---

## Output Files

| File | Description |
|:---|:---|
| `kerr_geodesic.png` | 3D trajectory for circular orbit |
| `conservation_check.png` | Hamiltonian conservation (H = -0.5) |
| `carter_check.png` | Carter constant conservation |
| `equatorial_section.png` | Equatorial plane projection |
| `kerr_plunging.png` | Plunging orbit 3D |
| `kerr_penrose.png` | Penrose process 3D |

---

## Numerical Method

- **ODE Solver**: SciPy `solve_ivp` with RK45 method
- **Derivatives**: Numerical differentiation (finite differences) of the Hamiltonian
- **JIT Compilation**: Numba for metric computations (near-C speed)
- **Tolerances**: rtol=1e-10, atol=1e-12 for high precision
- **Event Detection**: Automatic termination at event horizon

---

## Validation

The engine is validated against:
1. **EinsteinPy** — Open-source GR library
2. **Hamiltonian Conservation** — H must stay at -0.5
3. **Carter Constant** — Fourth integral of Kerr motion
4. **Energy Conservation** — p_t must be invariant
5. **Horizon Detection** — Correct event termination

---

## References

- **EinsteinPy**: [einsteinpy.org](https://einsteinpy.org)
- **Carroll GR Notes**: [preposterousuniverse.com/grnotes](https://preposterousuniverse.com/grnotes/)
- **MIT OCW 8.962**: [ocw.mit.edu](https://ocw.mit.edu)
- **EHT M87\* Data**: [eventhorizontelescope.org](https://eventhorizontelescope.org)
