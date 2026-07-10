# References — Annotated Bibliography

[← Documentation index](README.md)

Every entry below carries a stable HTML anchor of the form `#ref-<firstauthor>-<year>`, so other documentation pages (and the README) can deep-link to individual citations, e.g. [`references.md#ref-fu-kane-2008`](#ref-fu-kane-2008). Annotations state what each work established and which module or constant in this repository uses it. Several entries that the README cites by PMC number or with an incorrect author attribution have been resolved to their proper journal citations here; those entries carry a *Note* line documenting the correction.

Anchor quick-reference (for linking from other pages):

- **Primary papers**: `ref-wang-2026`, `ref-yan-2026`, `ref-kong-2025`
- **Foundations of superconductivity**: `ref-bardeen-1957`, `ref-abrikosov-1957`, `ref-blonder-1982`
- **Topological matter**: `ref-hasan-2010`, `ref-fu-kane-2008`, `ref-qi-2008`, `ref-sau-2010`, `ref-beenakker-2013`
- **Iron chalcogenides & isotope effects**: `ref-khasanov-2010`, `ref-liu-2009`, `ref-shirage-2009`, `ref-khasanov-2010b`, `ref-huang-2019`, `ref-song-2019`
- **NMR & nuclear spin**: `ref-ding-2025`, `ref-nachtigal-2022`
- **Phonon engineering**: `ref-bessas-2012`, `ref-weng-2021`, `ref-lindsay-2008`, `ref-lindsay-2013`
- **Graphene & twistronics**: `ref-castro-neto-2009`, `ref-bistritzer-macdonald-2011`, `ref-koshino-2018`, `ref-tarnopolsky-2019`, `ref-khalaf-2019`, `ref-cao-2018`, `ref-cao-2018b`, `ref-park-2021`, `ref-huder-2018`, `ref-wang-2019`
- **Strain & pseudo-magnetic fields**: `ref-vozmediano-2010`, `ref-guinea-2010`, `ref-levy-2010`, `ref-meyer-2007`, `ref-blakslee-1970`

## Primary papers

The two 2026 companion papers below are the experimental basis of this project. They postdate most of the literature in this bibliography; everything this documentation says about them traces to this repository's own description (README and the in-app About panel), not to an independent reading of the papers.

<a id="ref-wang-2026"></a>
**Z. Wang, B. Xia, S. Paolini, Z.-J. Yan, P. Xiao, J. Song, V. Gowda, H. Rong, D. Xiao, X. Xu, W. Wu, Z. Wang, and C.-Z. Chang**, "Moire Engineering of Cooper-Pair Density Modulation States," Nature 652, 335 (2026); [arXiv:2602.22637](https://arxiv.org/abs/2602.22637).

The paper this entire tool models: a moire superlattice formed by a topological-insulator film on FeTe imprints a periodic modulation on the superconducting gap. It fixes the repository's central defaults — the gap values `DELTA_1` = 2.58 meV and `DELTA_2` = 3.60 meV in `src/waytogocoop/config.py` — and drives the pattern/gap pipeline in `src/waytogocoop/computation/moire.py` (`generate_moire_pattern`) and `src/waytogocoop/computation/superconducting.py` (`gap_modulation`), mirrored in `crates/moire-core/src/`.

### What the primary paper reports

- Scanning tunneling microscopy/spectroscopy (STM/STS) reveals two distinct superconducting gaps, $\Delta_1 \approx 2.58$ meV and $\Delta_2 \approx 3.60$ meV, and both undergo periodic spatial modulation synchronized with the moire superlattice — the Cooper-pair density rises and falls with a wavelength matching the moire periodicity.
- Unlike a pair-density wave (PDW), which spontaneously breaks translational symmetry, the CPDM state inherits its modulation from an external periodic potential: the moire superlattice.
- The heterostructure is a single quintuple layer (1 QL) of the topological insulator Sb2Te3 grown epitaxially on a six-unit-cell-thick (6 UC) FeTe film.
- The hexagonal Te sublattice of Sb2Te3 (a ~ 4.26 angstrom) interferes with the square Te sublattice of FeTe (a ~ 3.82 angstrom) — two *different* crystal symmetries — producing a rhombic moire superlattice.
- Replacing Sb2Te3 with Bi2Te3 (a ~ 4.38 angstrom) changes the lattice mismatch, altering the moire periodicity and weakening the CPDM magnitude — demonstrating that the CPDM state is tunable by materials choice.

Summary based on this repository's description of the preprint.

<a id="ref-yan-2026"></a>
**Z.-J. Yan, Z. Wang, B. Xia, et al.**, "Stoichiometric FeTe is a superconductor," Nature 652, 342 (2026); [arXiv:2603.16115](https://arxiv.org/abs/2603.16115).

Companion paper in the same Nature issue (same collaboration): bulk FeTe is antiferromagnetic and non-superconducting because of excess interstitial iron, but stoichiometric FeTe films superconduct with Tc ~ 13.5 K — the substrate premise of the whole heterostructure. It backs the FeTe entry in `src/waytogocoop/materials/database.py` and the Tc ~ 13.5 K quoted in the README materials table. *Note: earlier revisions of the README misattributed this paper to "C. C. Homes et al."; the correct attribution is Yan et al.*

<a id="ref-kong-2025"></a>
**L. Kong et al.**, "Cooper-pair density modulation state in an iron-based superconductor," Nature 640, 55 (2025). [DOI: 10.1038/s41586-025-08703-x](https://doi.org/10.1038/s41586-025-08703-x)

The immediate predecessor of the primary paper: reported a Cooper-pair density modulation in Fe(Te,Se) and established the CPDM terminology — a gap modulation that does *not* spontaneously break translational symmetry, the distinction the primary paper and this repository's README carry forward. (Attribution verified by title, venue, and DOI; this documentation does not draw experimental details from the paper itself.)

## Foundations of superconductivity

<a id="ref-bardeen-1957"></a>
**J. Bardeen, L. N. Cooper, and J. R. Schrieffer**, "Theory of Superconductivity," Phys. Rev. 108, 1175 (1957). [DOI: 10.1103/PhysRev.108.1175](https://doi.org/10.1103/PhysRev.108.1175)

Established the microscopic theory of superconductivity: phonon-mediated Cooper pairing, the gap equation, and the isotope scaling $T_c \propto M^{-\alpha}$ with $\alpha = 1/2$ in the simplest limit. The weak-coupling ratio $\Delta(0) = 1.764\,k_B T_c$ is `BCS_GAP_RATIO` in `src/waytogocoop/config.py`, used by `compute_flat_band_sc` in `src/waytogocoop/computation/graphene.py`; the isotope scaling underlies `compute_isotope_effects` in `src/waytogocoop/computation/isotope_effects.py`.

<a id="ref-abrikosov-1957"></a>
**A. A. Abrikosov**, "On the Magnetic Properties of Superconductors of the Second Group," Sov. Phys. JETP 5, 1174 (1957).

Predicted that type-II superconductors admit magnetic flux as a triangular lattice of quantized vortices. Basis for `vortex_lattice_period` and `generate_vortex_positions` in `src/waytogocoop/computation/magnetic.py` (Rust: `crates/moire-core/src/magnetic.rs`) and the vortex-marker overlay in `crates/moire-desktop/src/render/overlay.rs`.

<a id="ref-blonder-1982"></a>
**G. E. Blonder, M. Tinkham, and T. M. Klapwijk**, "Transition from metallic to tunneling regimes in superconducting microconstrictions: Excess current, charge imbalance, and supercurrent conversion," Phys. Rev. B 25, 4515 (1982). [DOI: 10.1103/PhysRevB.25.4515](https://doi.org/10.1103/PhysRevB.25.4515)

The BTK model of transport across a normal-superconductor interface, parametrized by an interface transparency. It motivates the `interface_transparency` field of `ProximityConfig` in `src/waytogocoop/computation/topological.py`, which scales the proximity-induced gap entering the 3D extension.

## Topological matter

**Status: these references inform the SPECULATIVE topological module.** The Majorana modes, phase diagrams, and Chern-number estimates in `src/waytogocoop/computation/topological.py` are simplified 3D extensions of 2D models, not validated for these heterostructures; the app tags their outputs (SPECULATIVE). Treat them as exploratory illustrations, not quantitative predictions.

<a id="ref-hasan-2010"></a><a id="ref-hasan-kane-2010"></a>
**M. Z. Hasan and C. L. Kane**, "Colloquium: Topological insulators," Rev. Mod. Phys. 82, 3045 (2010). [DOI: 10.1103/RevModPhys.82.3045](https://doi.org/10.1103/RevModPhys.82.3045)

The standard review of topological insulators, including the Bi2Se3/Bi2Te3/Sb2Te3 family and their symmetry-protected Dirac surface states. Background for the overlayer descriptions in `src/waytogocoop/materials/database.py` and for `dirac_dispersion` in `src/waytogocoop/computation/topological.py`.

<a id="ref-fu-kane-2008"></a>
**L. Fu and C. L. Kane**, "Superconducting Proximity Effect and Majorana Fermions at the Surface of a Topological Insulator," Phys. Rev. Lett. 100, 096407 (2008). [DOI: 10.1103/PhysRevLett.100.096407](https://doi.org/10.1103/PhysRevLett.100.096407)

Showed that a conventional superconductor proximity-coupled to a TI surface produces an effective p+ip-like state whose vortices bind Majorana zero modes — the concept behind `majorana_probability_density` and `topological_phase_index` in `src/waytogocoop/computation/topological.py` (a simplified, qualitative implementation). The repository's surface-state g-factor default `G_FACTOR_TSS` = 30 in `src/waytogocoop/config.py` is a model default within the 20–50 literature range noted there.

<a id="ref-qi-2008"></a>
**X.-L. Qi, T. L. Hughes, and S.-C. Zhang**, "Topological field theory of time-reversal invariant insulators," Phys. Rev. B 78, 195424 (2008). [DOI: 10.1103/PhysRevB.78.195424](https://doi.org/10.1103/PhysRevB.78.195424)

Established the axion electrodynamics of 3D topological insulators: a quantized magnetoelectric term with axion angle theta = pi, so an applied magnetic field induces a quantized surface polarization. This is the formula behind `topological_magnetoelectric_polarization` in `src/waytogocoop/computation/topological.py` (speculative module).

<a id="ref-sau-2010"></a>
**J. D. Sau, R. M. Lutchyn, S. Tewari, and S. Das Sarma**, "Generic New Platform for Topological Quantum Computation Using Semiconductor Heterostructures," Phys. Rev. Lett. 104, 040502 (2010). [DOI: 10.1103/PhysRevLett.104.040502](https://doi.org/10.1103/PhysRevLett.104.040502)

Generalized the Fu-Kane recipe to semiconductor/superconductor heterostructures with Zeeman splitting, establishing the ingredient list (spin-orbit coupling + superconductivity + magnetic field) scanned by `phase_diagram_sweep` in `src/waytogocoop/computation/topological.py` and shown on the `/phase` page.

<a id="ref-beenakker-2013"></a>
**C. W. J. Beenakker**, "Search for Majorana Fermions in Superconductors," Annu. Rev. Condens. Matter Phys. 4, 113 (2013); [arXiv:1112.1950](https://arxiv.org/abs/1112.1950).

Review of the experimental Majorana search and its pitfalls (trivial states can mimic Majorana signatures). Context for why the Majorana features in `src/waytogocoop/computation/topological.py` are labeled speculative rather than predictive.

## Iron chalcogenides & isotope effects

**Status: these references inform the SPECULATIVE isotope module.** No direct Te-isotope data exists for FeTe; the default exponent in `src/waytogocoop/computation/isotope_effects.py` is taken from the iron-pnictide consensus, and outputs are exploratory illustrations, not quantitative predictions. The isotope cost tiers quoted in the README (e.g. 54Fe at roughly 5–10 EUR/mg) are repository-originated order-of-magnitude estimates, not literature values.

<a id="ref-khasanov-2010"></a>
**R. Khasanov et al.**, "Iron isotope effect on the superconducting transition temperature and the crystal structure of FeSe1-x," New J. Phys. 12, 073024 (2010); [arXiv:1002.2510](https://arxiv.org/abs/1002.2510). [DOI: 10.1088/1367-2630/12/7/073024](https://doi.org/10.1088/1367-2630/12/7/073024)

Measured an unusually large iron isotope exponent, alpha_Fe = 0.81 ± 0.15, in FeSe1-x — the closest chemical relative of FeTe with isotope data. Quoted in the `src/waytogocoop/computation/isotope_effects.py` docstring and the upper end of the alpha ranges marked on the isotope panel (`src/waytogocoop/components/isotope_panel.py`).

<a id="ref-liu-2009"></a>
**R. H. Liu et al.**, "A large iron isotope effect in SmFeAsO1-xFx and Ba1-xKxFe2As2," Nature 459, 64 (2009).

Reported a substantial normal iron isotope effect (alpha_Fe ~ 0.35) in two iron-pnictide families, arguing phonons matter for the pairing. One of the literature anchors for the alpha range documented in `src/waytogocoop/computation/isotope_effects.py` and `src/waytogocoop/config.py`.

<a id="ref-shirage-2009"></a>
**P. M. Shirage et al.**, "Inverse Iron Isotope Effect on the Transition Temperature of the (Ba,K)Fe2As2 Superconductor," Phys. Rev. Lett. 103, 257003 (2009). [DOI: 10.1103/PhysRevLett.103.257003](https://doi.org/10.1103/PhysRevLett.103.257003)

Found alpha_Fe = -0.18 in the same compound where Liu et al. found a large positive value — an unresolved contradiction. This is why the isotope-exponent slider in the app admits negative values and why the module is speculative.

<a id="ref-khasanov-2010b"></a>
**R. Khasanov, M. Bendele, A. Bussmann-Holder, and H. Keller**, "Intrinsic and structural isotope effects in iron-based superconductors," Phys. Rev. B 82, 212505 (2010). [DOI: 10.1103/PhysRevB.82.212505](https://doi.org/10.1103/PhysRevB.82.212505)

Proposed that isotope substitution also changes the crystal structure, and that correcting for this reconciles the conflicting measurements to a consensus alpha ~ 0.35–0.4. Source of the model default `DEFAULT_ISOTOPE_EXPONENT` = 0.4 in `src/waytogocoop/config.py` (cited there as "PRB 82, 212505").

<a id="ref-huang-2019"></a>
**W.-M. Huang and H.-H. Lin**, "Anomalous isotope effect in iron-based superconductors," Sci. Rep. 9, 5547 (2019). [DOI: 10.1038/s41598-019-42041-z](https://doi.org/10.1038/s41598-019-42041-z)

Renormalization-group analysis concluding the ground state is a "phonon-dressed unconventional superconductor": electronic pairing dominates while phonons dress the interaction, so the isotope effect can be normal or inverse. Quoted in the `src/waytogocoop/computation/isotope_effects.py` docstring. *Note: earlier revisions of the README cited this entry as "PMC6447578" with a different author attribution; the PMC ID resolves to Huang & Lin as above.*

<a id="ref-song-2019"></a>
**Q. Song et al.**, "Evidence of cooperative effect on the enhanced superconducting transition temperature at the FeSe/SrTiO3 interface," Nat. Commun. 10, 758 (2019). [DOI: 10.1038/s41467-019-08560-z](https://doi.org/10.1038/s41467-019-08560-z)

Showed that the Tc enhancement of monolayer FeSe on SrTiO3 combines FeSe's intrinsic pairing with coupling to substrate phonons crossing the interface — precedent for the cross-interface phonon arguments behind the speculative isotope-engineering ideas in the README and `src/waytogocoop/computation/isotope_effects.py`. *Note: earlier revisions of the README cited this entry as "PMC6377624" under the title of a related but different paper; the PMC ID resolves to Song et al. as above.*

## NMR & nuclear spin

<a id="ref-ding-2025"></a>
**Q.-P. Ding, J. Schmidt, J. A. Moreno, S. L. Bud'ko, P. C. Canfield, and Y. Furukawa**, "Role of Nematic Fluctuations on Superconductivity in FeSe0.47Te0.53 Revealed by NMR under Pressure," Phys. Rev. Lett. 134, 226002 (2025); [arXiv:2505.11732](https://arxiv.org/abs/2505.11732).

77Se/125Te NMR under pressure indicating that nematic fluctuations, not antiferromagnetic ones, dominate the superconductivity of FeSe0.47Te0.53 near its nematic quantum critical point — demonstrating 125Te NMR as a local probe of Fe-chalcogenide superconductivity. Motivates the 125Te spin-fraction bookkeeping (`te_125_spin_fraction` in `src/waytogocoop/materials/isotopes.py`) and the README's Tier-3 "125Te-enriched FeTe" diagnostic strategy. *Note: earlier revisions of the README attributed this preprint to "S.-H. Park et al." with a paraphrased title; the correct first author is Q.-P. Ding.*

<a id="ref-nachtigal-2022"></a>
**Nachtigal et al.**, "125Te NMR study of the bulk of topological insulators Bi2Te3 and Sb2Te3," Z. Anorg. Allg. Chem. 648, e202200208 (2022). [DOI: 10.1002/zaac.202200208](https://doi.org/10.1002/zaac.202200208)

Bulk 125Te NMR characterization of exactly the two topological-insulator overlayers used in this project, establishing that the Te sites of Sb2Te3/Bi2Te3 are NMR-accessible. Context for the nuclear-spin-bath discussion around `te_125_spin_fraction` (`src/waytogocoop/materials/isotopes.py`) and the isotope panel's spin-fraction readout. *Note: earlier revisions of the README attributed this paper to "M. Y. Seyidov et al."; the 2022 Z. Anorg. Allg. Chem. paper matching this title is by Nachtigal and co-workers.*

## Phonon engineering

<a id="ref-bessas-2012"></a>
**D. Bessas et al.**, "Lattice dynamics in Bi2Te3 and Sb2Te3: Te and Sb density of phonon states," Phys. Rev. B 86, 224301 (2012). [DOI: 10.1103/PhysRevB.86.224301](https://doi.org/10.1103/PhysRevB.86.224301)

Element-resolved phonon densities of states for both TI overlayers, showing which vibrational modes the Te and Sb sublattices carry. Literature context for the per-element Debye temperatures and Gruneisen parameters tabulated in `src/waytogocoop/materials/isotopes.py` and used by `compute_isotope_effects`.

<a id="ref-weng-2021"></a>
**H.-K. Weng, A. Nagakubo, H. Watanabe, and H. Ogi**, "Phonon propagation in isotopic diamond superlattices," Phys. Rev. B 104, 054112 (2021). [DOI: 10.1103/PhysRevB.104.054112](https://doi.org/10.1103/PhysRevB.104.054112)

Demonstrated phonon engineering with isotopically layered diamond — mass contrast alone, with identical chemistry, alters phonon propagation. Proof of concept behind the README's isotope-enrichment tiers and the lattice/Debye-Waller terms in `src/waytogocoop/computation/isotope_effects.py`. *Note: earlier revisions of the README attributed this paper to "S. Koga et al."; the PRB 104, 054112 author list is Weng, Nagakubo, Watanabe, and Ogi.*

<a id="ref-lindsay-2008"></a><a id="ref-lindsay-broido-2008"></a>
**L. Lindsay and D. A. Broido**, "Three-phonon phase space and lattice thermal conductivity in semiconductors," J. Phys.: Condens. Matter 20, 165209 (2008). [DOI: 10.1088/0953-8984/20/16/165209](https://doi.org/10.1088/0953-8984/20/16/165209)

Connected the phase space available for three-phonon scattering to lattice thermal conductivity across semiconductors. Background for the phonon-coherence reasoning in the README's isotope discussion (no direct code use).

<a id="ref-lindsay-2013"></a>
**L. Lindsay, D. A. Broido, and T. L. Reinecke**, "Phonon-isotope scattering and thermal conductivity in materials with a large isotope effect: A first-principles study," Phys. Rev. B 88, 144306 (2013). [DOI: 10.1103/PhysRevB.88.144306](https://doi.org/10.1103/PhysRevB.88.144306)

First-principles treatment of how isotopic mass disorder scatters phonons, quantifying what isotope purification buys. Supports the README's argument that isotope disorder (e.g. natural Sb or Te mixtures) degrades phonon coherence. *Note: the README lists this and the 2008 paper as one numbered entry; they are disambiguated here.*

## Graphene & twistronics

The flat-band superconductivity estimates built on these references (`compute_flat_band_sc`, `filling_dome_factor` in `src/waytogocoop/computation/graphene.py`) are marked SPECULATIVE in the app — the Lorentzian twist dome and filling dome are illustrative ansatzes, not quantitative predictions.

<a id="ref-castro-neto-2009"></a>
**A. H. Castro Neto, F. Guinea, N. M. R. Peres, K. S. Novoselov, and A. K. Geim**, "The electronic properties of graphene," Rev. Mod. Phys. 81, 109 (2009). [DOI: 10.1103/RevModPhys.81.109](https://doi.org/10.1103/RevModPhys.81.109)

The standard reference for graphene band parameters (nearest-neighbor hopping t = 2.8 eV, Dirac cones). Source of `HBAR_VF_GRAPHENE` = 5.96 eV-angstrom in `src/waytogocoop/config.py` — deliberately derived from t rather than the common 6.58 value, so the first magic angle lands in the observed 1.0–1.2 degree window (see the comment there).

<a id="ref-bistritzer-macdonald-2011"></a>
**R. Bistritzer and A. H. MacDonald**, "Moire bands in twisted double-layer graphene," Proc. Natl. Acad. Sci. USA 108, 12233 (2011). [DOI: 10.1073/pnas.1108174108](https://doi.org/10.1073/pnas.1108174108)

The continuum model of twisted bilayer graphene: interlayer tunneling folds the Dirac cones into moire bands whose velocity vanishes at the first magic angle (about 1.05–1.1 degrees). Implemented in `src/waytogocoop/computation/bm_model.py` (`build_bm_hamiltonian`, `compute_band_structure`; Rust: `crates/moire-core/src/bm_model.rs`) and, in closed form, as `magic_angle_deg` / `dirac_velocity_ratio` in `src/waytogocoop/computation/graphene.py`; sets `W_INTERLAYER_TBG` = 0.110 eV in `src/waytogocoop/config.py`.

<a id="ref-koshino-2018"></a>
**M. Koshino, N. F. Q. Yuan, T. Koretsune, M. Ochi, K. Kuroki, and L. Fu**, "Maximally Localized Wannier Orbitals and the Extended Hubbard Model for Twisted Bilayer Graphene," Phys. Rev. X 8, 031087 (2018). [DOI: 10.1103/PhysRevX.8.031087](https://doi.org/10.1103/PhysRevX.8.031087)

Showed that lattice corrugation shrinks AA-region tunneling relative to AB (w_AA < w_AB), which opens the gaps isolating the flat bands. Source of the defaults `W_AA_TBG` = 0.0797 eV and `W_AB_TBG` = 0.0975 eV in `src/waytogocoop/config.py`, used by the BM model in both languages.

<a id="ref-tarnopolsky-2019"></a>
**G. Tarnopolsky, A. J. Kruchkov, and A. Vishwanath**, "Origin of Magic Angles in Twisted Bilayer Graphene," Phys. Rev. Lett. 122, 106405 (2019). [DOI: 10.1103/PhysRevLett.122.106405](https://doi.org/10.1103/PhysRevLett.122.106405)

Identified the chiral limit w_AA = 0, where the flat bands become exactly flat with exact particle-hole symmetry. That symmetry is the cross-language test anchor `test_chiral_limit_particle_hole_symmetry` in `tests/test_bm_model.py` and `crates/moire-core/src/bm_model.rs`.

<a id="ref-khalaf-2019"></a>
**E. Khalaf, A. J. Kruchkov, G. Tarnopolsky, and A. Vishwanath**, "Magic angle hierarchy in twisted graphene multilayers," Phys. Rev. B 100, 085109 (2019). [DOI: 10.1103/PhysRevB.100.085109](https://doi.org/10.1103/PhysRevB.100.085109)

Mapped alternating-twist multilayers onto rescaled bilayer problems: the trilayer magic angle is sqrt(2) times the bilayer one. Implemented in `magic_angle_deg` in `src/waytogocoop/computation/graphene.py`, giving the approximately 1.52-degree magic angle for the `alternating_trilayer` stacking.

<a id="ref-cao-2018"></a>
**Y. Cao, V. Fatemi, S. Fang, K. Watanabe, T. Taniguchi, E. Kaxiras, and P. Jarillo-Herrero**, "Unconventional superconductivity in magic-angle graphene superlattices," Nature 556, 43 (2018). [DOI: 10.1038/nature26160](https://doi.org/10.1038/nature26160)

Discovered superconductivity (Tc ~ 2 K, near filling nu ~ 2.4) in magic-angle twisted bilayer graphene, launching twistronics as a field. Source of the speculative model inputs `DELTA_TBG_MAX` = 0.30 meV, `NU_OPTIMAL_FILLING` = 2.4, and `XI_TBG` = 500 angstrom in `src/waytogocoop/config.py`, used by `compute_flat_band_sc`.

<a id="ref-cao-2018b"></a>
**Y. Cao et al.**, "Correlated insulator behaviour at half-filling in magic-angle graphene superlattices," Nature 556, 80 (2018). [DOI: 10.1038/nature26154](https://doi.org/10.1038/nature26154)

Companion paper reporting correlated insulating states at half-filling of the flat bands — the reason superconductivity in TBG depends on band filling. Context for the filling-dependent dome `filling_dome_factor` in `src/waytogocoop/computation/graphene.py`.

<a id="ref-park-2021"></a>
**J. M. Park, Y. Cao, K. Watanabe, T. Taniguchi, and P. Jarillo-Herrero**, "Tunable strongly coupled superconductivity in magic-angle twisted trilayer graphene," Nature 590, 249 (2021).

Extended magic-angle superconductivity to alternating-twist trilayer graphene with Tc ~ 2.9 K, confirming the Khalaf et al. hierarchy. Source of `DELTA_TTG_MAX` = 0.44 meV in `src/waytogocoop/config.py` and of the `alternating_trilayer` preset in `src/waytogocoop/computation/graphene.py`.

<a id="ref-huder-2018"></a>
**L. Huder et al.**, "Electronic Spectrum of Twisted Graphene Layers under Heterostrain," Phys. Rev. Lett. 120, 156405 (2018). [DOI: 10.1103/PhysRevLett.120.156405](https://doi.org/10.1103/PhysRevLett.120.156405)

Showed that uniaxial strain of one layer relative to the other (heterostrain) reshapes the moire lattice and the low-energy spectrum. Basis for `strained_g_vectors` and the heterostrain arguments of `generate_stack_pattern_v2` in `src/waytogocoop/computation/graphene.py`, which strain one layer's reciprocal vectors before the twist.

<a id="ref-wang-2019"></a>
**Z. Wang et al.**, "Composite super-moire lattices in double-aligned graphene heterostructures," Sci. Adv. 5, eaay8897 (2019); [arXiv:1912.12268](https://arxiv.org/abs/1912.12268). [DOI: 10.1126/sciadv.aay8897](https://doi.org/10.1126/sciadv.aay8897)

Demonstrated that two coexisting moire patterns beat against each other to form a longer-period super-moire lattice. Basis for `generate_supermoire_pattern` in `src/waytogocoop/computation/graphene.py`, which stacks a graphene moire against a substrate overlayer and reports the beat period.

## Strain & pseudo-magnetic fields

The pseudo-magnetic field itself is established physics; the `gap_suppression_factor` pair-breaking ansatz built on it in `src/waytogocoop/computation/curvature.py` is SPECULATIVE (the valley-antisymmetric pseudo-field preserves time-reversal symmetry and does not orbitally depair singlet pairs) — treat those outputs as exploratory illustrations, not quantitative predictions.

<a id="ref-vozmediano-2010"></a>
**M. A. H. Vozmediano, M. I. Katsnelson, and F. Guinea**, "Gauge fields in graphene," Phys. Rep. 496, 109 (2010).

Review of how lattice deformations enter graphene's Dirac Hamiltonian as gauge fields, with the electron-phonon coupling beta = -dln(t)/dln(a) ~ 2–3.4. Source of `GRAPHENE_BETA` = 3.0 in `src/waytogocoop/config.py` and of the strain-to-gauge-field map in `pseudo_magnetic_field` (`src/waytogocoop/computation/curvature.py`).

<a id="ref-guinea-2010"></a>
**F. Guinea, M. I. Katsnelson, and A. K. Geim**, "Energy gaps and a zero-field quantum Hall effect in graphene by strain engineering," Nat. Phys. 6, 30 (2010).

Proposed engineering nearly uniform pseudo-magnetic fields with designed strain patterns — the "strain engineering" program. Motivates the designed height-field geometries (`height_field` in `src/waytogocoop/computation/curvature.py`: bump, ripple, bend, cap) whose pseudo-fields the `/graphene` curvature modes display.

<a id="ref-levy-2010"></a>
**N. Levy et al.**, "Strain-Induced Pseudo-Magnetic Fields Greater Than 300 Tesla in Graphene Nanobubbles," Science 329, 544 (2010).

Observed Landau-level-like spectra in graphene nanobubbles corresponding to pseudo-fields above 300 T — the experimental magnitude benchmark for the `gaussian_bump` geometry in `src/waytogocoop/computation/curvature.py` and its `max_abs_field` readout.

<a id="ref-meyer-2007"></a>
**J. C. Meyer, A. K. Geim, M. I. Katsnelson, K. S. Novoselov, T. J. Booth, and S. Roth**, "The structure of suspended graphene sheets," Nature 446, 60 (2007).

Found that free-standing graphene is intrinsically rippled on the nanometre scale rather than atomically flat. Source of the ripple defaults `RIPPLE_HEIGHT_DEFAULT` = 2 angstrom and `RIPPLE_WAVELENGTH_DEFAULT` = 100 angstrom in `src/waytogocoop/config.py`, used by the `sinusoidal_ripple` geometry.

<a id="ref-blakslee-1970"></a>
**O. L. Blakslee, D. G. Proctor, E. J. Seldin, G. B. Spence, and T. Weng**, "Elastic Constants of Compression-Annealed Pyrolytic Graphite," J. Appl. Phys. 41, 3373 (1970). [DOI: 10.1063/1.1659428](https://doi.org/10.1063/1.1659428)

Classic measurement of graphite's elastic constants, still the standard source for the in-plane Poisson ratio of graphene. Source of `POISSON_GRAPHENE` = 0.16 in `src/waytogocoop/config.py`, used by `strained_g_vectors` in `src/waytogocoop/computation/graphene.py` to contract the transverse axis under heterostrain.
