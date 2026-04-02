# Good Job Coop!

**Moire Engineering of Cooper-Pair Density Modulation States**

A visualization and computation tool for exploring moire-induced Cooper-pair density modulation in topological-insulator / iron-chalcogenide heterostructures.

Two implementations are provided:
- **Python Web UI** (Plotly Dash) -- accessible via browser on any platform
- **Rust Native App** (egui/eframe) -- native desktop for Linux, macOS, and Windows

Based on [Moire Engineering of Cooper-Pair Density Modulation States](https://arxiv.org/abs/2602.22637) (Wang, Xia, Paolini et al., 2026).

---

## What This Software Does

Good Job Coop analyzes how superconducting states are spatially modulated when a thin topological insulator film is grown on an iron-chalcogenide substrate. The lattice mismatch between the two layers creates a moire superlattice -- a long-wavelength interference pattern -- that imprints a periodic variation onto the superconducting Cooper-pair density. This tool lets you predict, visualize, and explore those modulations for different material combinations.

In practice, you select a substrate and an overlayer material, optionally set a twist angle, and the software computes:

1. **The moire superlattice geometry** -- real-space interference pattern from the two overlaid crystal lattices
2. **The moire periodicity** -- the characteristic wavelength of the emerging pattern
3. **The Cooper-pair density modulation (CPDM)** -- how the superconducting gap varies spatially across the moire unit cell
4. **The Fourier spectrum** -- reciprocal-space analysis revealing the dominant modulation wavevectors

Results are displayed as interactive heatmaps, line plots, and parameter sweeps that update in real time as you adjust material parameters.

### Concrete Examples by Substrate Combination

All examples below use FeTe (a = 3.82 angstrom, square lattice) as the substrate, which is the superconducting platform in each heterostructure.

#### Sb2Te3 / FeTe -- the primary system

- **Overlayer**: Sb2Te3 (a = 4.264 angstrom, hexagonal)
- **Lattice mismatch**: 11.6%
- **Moire periodicity**: ~36.7 angstrom (~3.7 nm)
- **Observed CPDM**: Two superconducting gaps (Delta_1 ~ 2.58 meV, Delta_2 ~ 3.60 meV) both modulated at the moire wavelength. This is the strongest CPDM signal among the studied systems.
- **Use case**: Baseline configuration. Select "Sb2Te3" as overlayer and "FeTe" as substrate with zero twist angle to reproduce the moire pattern observed in STM experiments. Use the parameter sweep to explore how small twist angles would change the periodicity, or the Fourier page to identify the moire reciprocal lattice vectors.

#### Bi2Te3 / FeTe -- tuned for longer periodicity

- **Overlayer**: Bi2Te3 (a = 4.386 angstrom, hexagonal)
- **Lattice mismatch**: 14.8%
- **Moire periodicity**: ~29.6 angstrom (~3.0 nm)
- **Observed CPDM**: Weaker modulation amplitude than Sb2Te3/FeTe, demonstrating that CPDM strength can be tuned by swapping the overlayer material. The shorter moire period relative to the superconducting coherence length reduces the modulation depth.
- **Use case**: Compare side-by-side with Sb2Te3/FeTe to see how a larger lattice mismatch shrinks the moire period and weakens the density modulation. The parameter sweep page is particularly useful here -- sweep the overlayer lattice constant from 4.26 to 4.39 angstrom to see the continuous transition between the two systems.

#### Sb2Te / FeTe -- intermediate variant

- **Overlayer**: Sb2Te (a = 4.272 angstrom, hexagonal)
- **Lattice mismatch**: 11.8%
- **Moire periodicity**: ~36.1 angstrom (~3.6 nm)
- **Expected CPDM**: Very close to the Sb2Te3/FeTe system due to the nearly identical lattice constant (4.272 vs. 4.264 angstrom). The slightly larger mismatch yields a marginally shorter moire period.
- **Use case**: Explore the sensitivity of the moire pattern to small changes in overlayer lattice constant. Switching between Sb2Te3 and Sb2Te in the viewer shows how a 0.008 angstrom change in the overlayer spacing shifts the moire periodicity by ~0.6 angstrom -- a demonstration of the fine tunability available through material selection.

#### Custom configurations

Beyond the built-in presets, both the web UI and desktop app allow you to override any lattice constant with a custom value. This enables exploration of hypothetical or strained heterostructures, for example:

- **Strained Sb2Te3**: Set the Sb2Te3 lattice constant to 4.10 angstrom (compressive strain) to see how the moire period increases to ~55.7 angstrom as the mismatch shrinks.
- **Twist-angle engineering**: Apply a 2-degree twist to the Sb2Te3/FeTe system and observe the moire pattern transition from a pure-mismatch regime to a twist-dominated regime where L ~ a/theta.
- **Matched lattices**: Set both layers to the same lattice constant to verify that the moire pattern vanishes (infinite periodicity), confirming the mismatch-driven origin.

---

## Table of Contents

- [What This Software Does](#what-this-software-does)
  - [Concrete Examples by Substrate Combination](#concrete-examples-by-substrate-combination)
- [Theory Background](#theory-background)
  - [Moire Patterns in Crystal Lattices](#moire-patterns-in-crystal-lattices)
  - [Cooper Pairs and Superconductivity](#cooper-pairs-and-superconductivity)
  - [Cooper-Pair Density Modulation](#cooper-pair-density-modulation-cpdm)
  - [Moire Engineering of CPDM States](#moire-engineering-of-cpdm-states)
  - [Materials](#materials)
  - [Key Equations](#key-equations)
- [Features](#features)
- [Installation and Usage](#installation-and-usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

---

## Theory Background

### Moire Patterns in Crystal Lattices

When two periodic lattices are overlaid with a slight mismatch -- either a difference in lattice spacing or a relative twist angle -- an emergent long-wavelength pattern appears called a *moire pattern*. In crystalline materials, a moire superlattice arises whenever two atomic layers with differing periodicities are stacked. The moire pattern has a much larger periodicity than either constituent lattice, and this new length scale can profoundly alter the electronic properties of the combined system.

A particularly interesting situation occurs when the two layers have *different crystal symmetries*. In the heterostructures studied here, a hexagonal Te sublattice (from a topological insulator such as Sb2Te3) sits atop a square Te sublattice (from FeTe). The geometric interference between these two incompatible symmetries generates a rhombic moire superlattice whose periodicity is governed by the lattice mismatch between the layers.

### Cooper Pairs and Superconductivity

Superconductivity is a quantum state of matter in which electrical resistance drops to exactly zero below a critical temperature T_c. The microscopic origin, as described by Bardeen-Cooper-Schrieffer (BCS) theory, involves the formation of *Cooper pairs*: bound states of two electrons with opposite momenta and spins, mediated by lattice vibrations (phonons) or other bosonic excitations. These pairs condense into a macroscopic quantum state described by a complex order parameter Delta (the superconducting gap), whose magnitude measures the binding energy of the Cooper pairs and whose phase is coherent across the entire superconductor.

In a conventional homogeneous superconductor, the gap Delta is spatially uniform. However, certain exotic superconducting phases exhibit spatial modulation of the order parameter, meaning that the density of Cooper pairs varies periodically in real space.

### Cooper-Pair Density Modulation (CPDM)

A Cooper-pair density modulation (CPDM) state is a superconducting phase in which the superconducting order parameter varies periodically in real space. Unlike a pair-density wave (PDW) that spontaneously breaks translational symmetry, a CPDM state inherits its spatial modulation from an external periodic potential -- in this case, the moire superlattice.

In the Sb2Te3/FeTe heterostructure, scanning tunneling microscopy and spectroscopy (STM/STS) reveal two distinct superconducting gaps: Delta_1 ~ 2.58 meV and Delta_2 ~ 3.60 meV. Both gaps undergo periodic spatial modulation synchronized with the moire superlattice pattern. The Cooper-pair density rises and falls in a repeating pattern whose wavelength matches the moire periodicity.

### Moire Engineering of CPDM States

The key insight of moire engineering is that the CPDM wavelength and amplitude can be *tuned* by choosing different material combinations that change the lattice mismatch:

- A single quintuple layer (1 QL) of the topological insulator Sb2Te3 is epitaxially grown on a six-unit-cell-thick (6 UC) film of FeTe.
- The hexagonal Te sublattice of Sb2Te3 (a ~ 4.26 angstrom) interferes with the square Te sublattice of FeTe (a ~ 3.82 angstrom), creating a rhombic moire superlattice.
- When Sb2Te3 is replaced by Bi2Te3 (a ~ 4.38 angstrom), the different lattice mismatch produces a moire pattern with altered periodicity and a weaker CPDM magnitude, demonstrating tunability.

This epitaxial strategy for synthesizing moire superlattices from materials with different crystal symmetries reveals a mechanism for engineering CPDM states in designer bilayer heterostructures.

### Materials

| Material | Role | Crystal Symmetry | In-plane lattice constant *a* |
|----------|------|------------------|-------------------------------|
| FeTe     | Antiferromagnetic substrate (T_c ~ 13.5 K when stoichiometric) | Tetragonal (P4/nmm), square Te sublattice | ~3.82 angstrom |
| Sb2Te3   | Topological insulator overlayer | Rhombohedral (R-3m), hexagonal Te sublattice | ~4.264 angstrom |
| Bi2Te3   | Alternative TI for tuning moire periodicity | Rhombohedral (R-3m), hexagonal Te sublattice | ~4.386 angstrom |
| Sb2Te    | Related Sb-Te binary compound | Rhombohedral, hexagonal sublattice | ~4.272 angstrom |

**FeTe** is the parent compound of the iron-chalcogenide superconductor family. Bulk FeTe is antiferromagnetic and non-superconducting due to excess interstitial iron atoms. When grown as a thin film and the excess iron is removed (e.g., by annealing in Te vapor), stoichiometric FeTe exhibits superconductivity with T_c ~ 13.5 K.

**Sb2Te3** and **Bi2Te3** are three-dimensional topological insulators: they are insulating in the bulk but host conducting surface states protected by time-reversal symmetry. A single quintuple layer (QL) consists of five atomic planes stacked as Te-Sb-Te-Sb-Te (or Te-Bi-Te-Bi-Te), held together by covalent bonds within the layer and van der Waals forces between layers.

### Key Equations

**Moire periodicity from a twist angle** (same lattice constant, relative rotation by angle theta):

```
L = a / (2 * sin(theta / 2))
```

For small angles, this simplifies to L ~ a / theta.

**Moire periodicity from lattice mismatch at zero twist angle** (two different lattice constants, no rotation):

```
L = (a1 * a2) / |a1 - a2|
```

For example, using a1 = 4.264 angstrom (Sb2Te3) and a2 = 3.82 angstrom (FeTe):

```
L = (4.264 * 3.82) / |4.264 - 3.82| = 16.29 / 0.444 ~ 36.7 angstrom ~ 3.7 nm
```

**Note:** The actual moire geometry in the Sb2Te3/FeTe system is more complex because the two lattices have different symmetries (hexagonal vs. square). The simple 1D formula provides an approximate scale. The full 2D moire pattern is rhombic, described by commensurate supercell vectors.

**Gap modulation model:**

```
Delta(r) = Delta_avg + delta_Delta * cos(Q_moire . r + phi)
```

where Q_moire are the moire reciprocal vectors, delta_Delta is the modulation amplitude, and phi is the phase shift relative to the moire potential.

---

## Features

- **Moire pattern visualization**: Real-space rendering of the moire superlattice formed by overlaying hexagonal and square lattices with configurable lattice constants.
- **Periodicity calculator**: Compute moire periodicity from lattice constants (a1, a2) and twist angle (theta).
- **Material presets**: Built-in lattice parameters for Sb2Te3, Bi2Te3, FeTe, and Sb2Te.
- **Gap modulation profile**: Visualize how the two superconducting gaps (Delta_1, Delta_2) are spatially modulated across the moire unit cell.
- **Interactive parameter tuning**: Adjust lattice constants, twist angle, and gap parameters with sliders and see results update in real time.
- **Fourier analysis**: 2D FFT power spectrum revealing moire reciprocal lattice vectors.
- **Parameter sweeps**: Explore how moire periodicity and CPDM amplitude vary with lattice constant or twist angle.

---

## Installation and Usage

### Python Web UI (Plotly Dash)

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/user/waytogocoop.git
cd waytogocoop

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install the package
pip install -e .

# Run the web UI
python -m waytogocoop.app
```

The web UI will be available at `http://localhost:8050`.

### Rust Native App (egui/eframe)

**Requirements:** Rust toolchain ([rustup](https://rustup.rs/))

```bash
cd waytogocoop

# Build and run
cargo run --release -p moire-desktop
```

Supported platforms: Linux (X11/Wayland), macOS, Windows.

---

## Project Structure

```
waytogocoop/
  README.md
  CLAUDE.md
  pyproject.toml                    # Python project config
  Cargo.toml                       # Rust workspace root

  src/waytogocoop/                  # Python Web UI
    app.py                          # Dash entry point
    materials/                      # Material database + lattice generation
    computation/                    # Moire, superconducting, FFT modules
    pages/                          # Dash pages (viewer, sweep, fourier)
    components/                     # Reusable UI components

  crates/
    moire-core/                     # Rust computation library
      src/                          # lattice, materials, moire, density, fft
    moire-desktop/                  # Rust egui desktop app
      src/                          # app, ui/, render/

  tests/                            # Python tests
```

---

## References

1. Z. Wang, B. Xia, S. Paolini, Z.-J. Yan, P. Xiao, J. Song, V. Gowda, H. Rong, D. Xiao, X. Xu, W. Wu, Z. Wang, and C.-Z. Chang, "Moire Engineering of Cooper-Pair Density Modulation States," [arXiv:2602.22637](https://arxiv.org/abs/2602.22637) (2026).

2. J. Bardeen, L. N. Cooper, and J. R. Schrieffer, "Theory of Superconductivity," Phys. Rev. 108, 1175 (1957).

3. M. Z. Hasan and C. L. Kane, "Colloquium: Topological insulators," Rev. Mod. Phys. 82, 3045 (2010).

---

## License

This project is provided for research and educational purposes.
