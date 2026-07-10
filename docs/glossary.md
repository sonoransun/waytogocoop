# Glossary

[← Documentation index](README.md)

Terms are defined **as used in this repository**. Numeric defaults in parentheses come from `src/waytogocoop/config.py` (mirrored in the Rust crates) and are **model inputs with literature ranges, not measured values for this heterostructure**. Entries tied to speculative modules say so explicitly — as the in-app About panel puts it, speculative module outputs carry a `(SPECULATIVE)` tag in the title; treat them as exploratory illustrations, not quantitative predictions.

## Moire & lattices

**G-vector** — A reciprocal-lattice vector of one of the stacked layers. The moire potential is built as a plane-wave superposition $V(\mathbf{r}) = \sum_i \cos(\mathbf{G}_i \cdot \mathbf{r})$ over each layer's first G-shell: 4 vectors for a square lattice, 6 at 60-degree intervals for a hexagonal one (`src/waytogocoop/computation/moire.py`, `generate_moire_pattern`). See [Moire patterns](theory/moire-patterns.md).

**Moire Brillouin zone** — The small Brillouin zone of the moire superlattice, with characteristic wavevector $k_\theta = 2 k_D \sin(\theta/2)$ for twisted layers. BM band structures are plotted along its K → Γ → M → K′ path (`src/waytogocoop/computation/bm_model.py`, `compute_band_structure`). See [BM model](theory/bm-model.md).

**Moire superlattice** — The long-wavelength interference pattern that emerges when two lattices with slightly different spacings or a relative twist are overlaid. In Sb2Te3/FeTe, a hexagonal Te sublattice on a square Te sublattice yields a rhombic moire superlattice with period ~36.7 Å; the period follows $L = a_1 a_2 / |a_1 - a_2|$ (mismatch-driven) or $L = a / (2\sin(\theta/2))$ (twist-driven). See [Moire patterns](theory/moire-patterns.md).

**Quintuple layer (QL)** — The structural unit of the Sb2Te3/Bi2Te3 family of topological insulators: five atomic planes (Te-Sb-Te-Sb-Te) covalently bonded within the layer, van der Waals coupled between layers. The reference heterostructure uses a single QL of Sb2Te3 ([Wang et al. 2026](references.md#ref-wang-2026)).

**Unit cell (UC)** — One repeat unit of a crystal lattice, used here mainly as a film-thickness unit for the substrate: the reference heterostructure grows 1 QL Sb2Te3 on a 6 UC FeTe film.

## Superconductivity

**BCS theory** — The Bardeen-Cooper-Schrieffer theory of phonon-mediated superconductivity. The repo uses its weak-coupling results: the gap ratio Delta(0)/(kB Tc) = 1.764 (`BCS_GAP_RATIO`) and the isotope scaling of gap and Tc with ionic mass. See [Gap modulation](theory/gap-modulation.md) and [Isotope effects](theory/isotope-effects.md).

**Coherence length (xi)** — The size of a Cooper pair, which sets how well the condensate can follow a spatial modulation: the CPDM amplitude scales as $e^{-\xi / L_{moire}}$ (`src/waytogocoop/computation/superconducting.py`, `cpdm_amplitude`). Model defaults: xi = 20 Å (BCS estimate for FeTe-based systems), xi = 500 Å for magic-angle graphene (`XI_TBG`).

**Cooper pair** — A bound pair of electrons that condenses, with all other pairs, into the superconducting state. The "Cooper-pair density" whose modulation this tool visualizes tracks the local gap Delta(r).

**Critical temperature (Tc)** — The temperature below which a material superconducts. Stoichiometric FeTe films show Tc ~ 13.5 K ([Yan et al. 2026](references.md#ref-yan-2026), as described in this repo's README); the speculative graphene flat-band module uses Tc ~ 2 K (bilayer, [Cao et al. 2018](references.md#ref-cao-2018)) and ~2.9 K (trilayer, Park 2021) as model inputs.

**Gap (Delta)** — The superconducting energy gap in the quasiparticle spectrum — the quantity STM/STS actually maps. Sb2Te3/FeTe shows two gaps, Delta_1 = 2.58 meV (surface-state) and Delta_2 = 3.60 meV (bulk-like), both modulated at the moire period: $\Delta(\mathbf{r}) = \Delta_{avg} + \delta\Delta \cos(\mathbf{Q}_{moire} \cdot \mathbf{r} + \varphi)$ (`superconducting.py`, `gap_modulation`). See [Gap modulation](theory/gap-modulation.md).

**London penetration depth (lambda_L)** — The depth over which supercurrents screen a magnetic field. Model default lambda_L = 5000 Å (~500 nm, typical for iron chalcogenides); it sets the decay envelope of the screening-current plots. See [Magnetic field](theory/magnetic.md).

**Order parameter** — The complex field $\Delta(\mathbf{r}) = |\Delta| e^{i\varphi}$ describing the superconducting condensate; its magnitude is the gap. A CPDM is by definition a state in which this order parameter varies periodically in real space.

**Type-II superconductor** — A superconductor that admits quantized flux vortices over a field range up to Bc2 instead of expelling the field entirely. The FeTe-based systems modeled here are strongly type-II (model Bc2 = 47 T).

## Modulated pairing

**Cooper-pair density modulation (CPDM)** — A superconducting phase whose order parameter varies periodically in real space because the modulation is *inherited from an external periodic potential* — here, the moire superlattice imprints it. Unlike a PDW there is no spontaneous breaking of translational symmetry; the wavelength is designed in via the material pair. The central subject of the primary paper ([Wang et al. 2026](references.md#ref-wang-2026)). See [Gap modulation](theory/gap-modulation.md).

**FFLO state** — The Fulde-Ferrell-Larkin-Ovchinnikov state (1964–65), the original theoretical modulated superconductor: a strong Zeeman field drives pairing at finite momentum, so the order parameter oscillates spontaneously. The historical ancestor of PDW/CPDM physics; not directly modeled here. See [History](history.md).

**Pair-density wave (PDW)** — A superconducting state whose order parameter modulates in space by *spontaneously* breaking translational symmetry — no external potential selects the wavelength. Contrast CPDM, where the moire superlattice supplies the period. See [Gap modulation](theory/gap-modulation.md).

## Magnetic field

**Abrikosov vortex** — A quantized flux tube in a type-II superconductor; vortices self-organize into a triangular lattice with spacing proportional to $1/\sqrt{B_z}$ (~489 Å at 1 T). Core gap suppression follows a Ginzburg-Landau profile $\tanh(r/\xi)$ (`src/waytogocoop/computation/magnetic.py`, `vortex_suppression_field`). See [Magnetic field](theory/magnetic.md).

**Commensuration field** — The perpendicular field at which the vortex-lattice period matches the moire period, $B_{comm} = 2\Phi_0 / (\sqrt{3} L_m^2)$ (`magnetic.py`, `commensuration_field`). For Sb2Te3/FeTe (L_m ~ 36.7 Å) it exceeds 100 T, so the two lattices are strongly incommensurate at laboratory fields.

**Flux quantum** — $\Phi_0 = h/2e = 2.068 \times 10^{-15}$ Wb, the magnetic flux carried by one vortex. `flux_per_moire_cell` counts flux quanta threading each moire unit cell.

**g-factor** — Dimensionless factor converting field to Zeeman energy, $E_Z = g \mu_B |B_\parallel|$. Model default g = 30 for topological surface states (`G_FACTOR_TSS`), literature range 20–50.

**Meissner screening** — Field expulsion by circulating supercurrents. Modeled as azimuthal screening currents around each vortex core decaying over lambda_L (5000 Å) (`magnetic.py`, `screening_currents`), displayed as quiver plots.

**Pauli limit** — The in-plane field at which Zeeman splitting equals the pair-breaking energy, $B_P = \Delta / (\sqrt{2}\,\mu_B)$ (`magnetic.py`, `pauli_limiting_field`). The magnetic page reports the depairing ratio B_par / B_P.

**Upper critical field (Bc2)** — The field at which vortex cores overlap and superconductivity is destroyed. Model default Bc2 = 47 T for FeTe (`BC2_FETE`), a low-temperature literature value used as an input scale.

**Zeeman splitting** — Spin splitting $E_Z = g \mu_B |B_\parallel|$ from an in-plane field (`magnetic.py`, `zeeman_energy`); the key input to the Fu-Kane criterion below. See [Magnetic field](theory/magnetic.md).

## Topology

**Axion angle** — The angle theta in the topological magnetoelectric term; theta = pi in a TI gives a quantized surface polarization $P = (e^2 / 2\pi h)\,\theta B$ (Planck constant $h$) (`src/waytogocoop/computation/topological.py`, `topological_magnetoelectric_polarization`; speculative module). See [Topological](theory/topological.md).

**BTK transparency** — The interface transmission T in (0, 1] of the Blonder-Tinkham-Klapwijk picture; it sets the induced gap at the SC/TI interface before the exponential decay into the TI (`topological.py`, `proximity_decay_profile`; default T = 0.8).

**Chern number** — A topological invariant counting chiral edge modes. Estimated here per surface as $C = \mathrm{sign}(E_Z^2 - \Delta^2 - \mu^2)/2$, giving ±0.5 (`topological.py`, `chern_number_estimate`; speculative module).

**Dirac cone** — A linear band crossing $E = \pm \hbar v_F |k|$. The TI surface state hosts one (model v_F = 5 × 10^5 m/s, `V_F_TI`); each graphene valley hosts one (hbar v_F = 5.96 eV Å, `HBAR_VF_GRAPHENE`).

**Fu-Kane criterion** — The condition $E_Z > \sqrt{\Delta^2 + \mu^2}$ for a proximitized TI surface to enter the topological phase that can host Majorana modes ([Fu & Kane 2008](references.md#ref-fu-kane-2008)); implemented in simplified, qualitative form in `topological.py`, `topological_phase_index` (speculative module). See [Topological](theory/topological.md).

**Majorana zero mode** — A predicted zero-energy quasiparticle, its own antiparticle, localized at vortex cores of a proximitized TI surface. Modeled as $|\psi|^2 \sim e^{-2r/\xi_M} J_0(k_F r)^2$ with xi_M = 50 Å default (`topological.py`, `majorana_probability_density`; speculative module).

**Proximity effect** — A superconductor inducing a gap in an adjacent non-superconducting material. Modeled as the gap decaying into the TI as $T e^{-z/\xi_{prox}}$ with xi_prox = 100 Å default (literature 50–200 Å). See [Topological](theory/topological.md).

**Quantum anomalous Hall (QAH) effect** — Quantized Hall conductance at zero external field, arising from a nonzero Chern number; the repo's Chern estimate would manifest as a quantized anomalous Hall plateau (speculative module, contextual).

**Surface state** — The protected metallic boundary state of a 3D TI; in this heterostructure it is the state that acquires the proximity-induced gap Delta_1 = 2.58 meV.

**Topological insulator (TI)** — A material insulating in the bulk with conducting surface states protected by time-reversal symmetry — Sb2Te3, Bi2Te3, and Sb2Te in the materials database; the moire overlayer in these heterostructures. See [Materials](materials.md).

## Twistronics & strain

**Bistritzer-MacDonald (BM) model** — The continuum model of twisted bilayer graphene: two Dirac cones coupled on a momentum lattice by interlayer tunneling ([Bistritzer & MacDonald 2011](references.md#ref-bistritzer-macdonald-2011)). Implemented with defaults w_AA = 79.7 meV, w_AB = 97.5 meV, 3 momentum shells, 16 k-points per segment (`src/waytogocoop/computation/bm_model.py`). See [BM model](theory/bm-model.md).

**Chiral limit** — The BM model with w_AA = 0, where the spectrum has exact particle-hole symmetry and the flat bands become exactly flat at the magic angle (Tarnopolsky et al. 2019). Used as a cross-language test anchor in both the Python and Rust suites.

**Flat band** — A nearly dispersionless moire band near the magic angle; the renormalized velocity is $v^*/v = (1 - 3\alpha^2)/(1 + 6\alpha^2)$ with $\alpha = w/(\hbar v_F k_\theta)$. Flat bands quench kinetic energy and enable correlated states including superconductivity ([Cao et al. 2018](references.md#ref-cao-2018)).

**Heterostrain** — Uniaxial strain applied to one layer relative to the other; a fraction-of-a-percent atomic strain distorts the moire pattern dramatically. Applied to one layer's G-vectors with graphene Poisson ratio 0.16 (`src/waytogocoop/computation/graphene.py`, `strained_g_vectors`). See [Graphene stacks](theory/graphene-stacks.md).

**Magic angle** — The twist angle at which the flat-band velocity vanishes: theta ~ 1.08 deg for bilayer graphene (with hbar v_F = 5.96 eV Å and w = 110 meV) and ~ 1.52 deg for alternating-twist trilayer (a factor sqrt(2) higher). See [Graphene stacks](theory/graphene-stacks.md).

**Monge gauge** — Describing a curved sheet by a height field h(x, y) with no in-plane relaxation, giving the strain tensor $\varepsilon_{ij} = \tfrac{1}{2} \partial_i h\, \partial_j h$ (`src/waytogocoop/computation/curvature.py`). The starting point of the pseudo-field calculation. See [Curvature](theory/curvature.md).

**Pseudo-magnetic field** — The effective field $B_{ps} = \nabla \times \mathbf{A}$ that strain produces for graphene's Dirac electrons; it can reach hundreds of tesla in nanobubbles. It is valley-antisymmetric (+B at K, −B at K′) and preserves time-reversal symmetry, so it does *not* orbitally depair singlet Cooper pairs — the repo's gap-suppression knob (`B_PAIRBREAK_DEFAULT` = 10 T) is explicitly a speculative ansatz. See [Curvature](theory/curvature.md).

**Supermoire** — The beating of two moire patterns — here the graphene-stack moire and the stack-on-substrate moire — producing a third, longer period computed as the 1D beat of the two (`graphene.py`, `generate_supermoire_pattern`).

**Twist angle (theta)** — The relative rotation between stacked layers (default 0 deg for the TI/FeTe system, whose moire is mismatch-driven). Twist-driven moire period: $L = a/(2\sin(\theta/2))$.

**Valley** — One of the two inequivalent Dirac points ±K of graphene's Brillouin zone. Curvature outputs are valley-resolved (valley parameter ±1) because the pseudo-magnetic field flips sign between valleys.

## Isotopes

*All entries below belong to the speculative isotope module — exploratory illustrations, not quantitative predictions. See [Isotope effects](theory/isotope-effects.md).*

**Debye temperature** — The temperature scale of a material's phonon spectrum, tabulated per element in `src/waytogocoop/materials/isotopes.py`; an input to the zero-point lattice-shift estimate.

**Debye-Waller factor** — The damping of diffraction or tunneling contrast by lattice vibrations. Heavier isotopes vibrate less, so enrichment sharpens the modeled STM contrast (`src/waytogocoop/computation/isotope_effects.py`).

**Gruneisen parameter** — The dimensionless anharmonicity measure gamma linking vibrational energy to lattice expansion; per-element values feed the zero-point lattice-shift formula (`isotope_effects.py`).

**Isotope exponent (alpha)** — The exponent in $T_c \propto M^{-\alpha}$ (BCS predicts 0.5). Model default alpha = 0.4, the corrected iron-chalcogenide consensus; literature spans 0.35–0.8 for iron chalcogenides, with an inverse effect alpha ~ −0.18 in (Ba,K)Fe2As2, and the slider covers [−0.5, 1.0].

**Nuclear spin bath** — The ensemble of nonzero-spin nuclei that produces magnetic noise and decoherence. 125Te (I = 1/2, 7.07% natural abundance) is the only spin-bearing stable Te isotope; enriching to 130Te eliminates the bath entirely, and the software reports the 125Te spin fraction for any enrichment. See [Materials](materials.md).

**Synthetic isotope** — A radioactive isotope with zero natural abundance, produced artificially; the database carries them with half-lives (e.g. 55Fe t½ ≈ 2.7 y, 60Fe ≈ 2.6 My, 14C ≈ 5700 y) for the HIGHLY SPECULATIVE exotic-isotope mode, which widens the mass sliders to include them and hypothetical masses beyond any known nuclide. Most of the listed half-lives are far too short to grow and measure a film, and the model ignores radioactivity entirely — outputs are what-if illustrations. See [Isotope effects](theory/isotope-effects.md).

**Zero-point motion** — The quantum vibrational amplitude that persists at T = 0. It is mass-dependent, so isotope substitution shifts the equilibrium lattice constant through anharmonic effects — the repo's speculative lattice-shift channel, $da = -a\, (3\gamma k_B T_D)/(4 E_{coh}) \, (1 - \sqrt{M_{nat}/M_{enr}})$.

## Experimental methods

**ARPES** — Angle-resolved photoemission spectroscopy, which measures band dispersion directly; the standard probe of TI Dirac cones. Contextual — not modeled here. See [Process technologies](process-technologies.md).

**dI/dV** — Differential tunneling conductance measured by STS, proportional to the local density of states. Fitting dI/dV spectra pixel by pixel yields the experimental Delta(r) maps that the viewer's gap heatmaps simulate.

**FT-STS** — Fourier-transform STS: taking the 2D FFT of dI/dV maps to reveal spatial periodicities in reciprocal space. The Fourier page (`src/waytogocoop/computation/fourier.py`, `fft_2d` and `identify_peaks`) is the simulation analog. See [Fourier analysis](theory/fourier-analysis.md).

**MBE** — Molecular-beam epitaxy: ultrahigh-vacuum, layer-by-layer film growth — the technique behind the 1 QL / 6 UC thickness control of the reference heterostructure. See [Process technologies](process-technologies.md).

**RHEED** — Reflection high-energy electron diffraction, the in-situ monitor of MBE growth; its intensity oscillations count atomic layers as they form.

**STM/STS** — Scanning tunneling microscopy/spectroscopy: atomic-scale imaging plus local tunneling spectra. The experiments this tool models used STM/STS to observe the moire pattern and the two modulated gaps (Delta_1 = 2.58 meV, Delta_2 = 3.60 meV) ([Wang et al. 2026](references.md#ref-wang-2026)).
