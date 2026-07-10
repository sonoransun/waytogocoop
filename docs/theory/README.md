# Theory Documentation

[← Documentation index](../README.md) · [Project README](../../README.md)

These pages are the physics deep-dives behind Good Job Coop. Each one expands a topic the
[project README](../../README.md) introduces: full model equations, derivations, assumptions
and validity limits, and pointers into the code. The physics is implemented twice — as pure
NumPy modules under `src/waytogocoop/computation/` (driving the Plotly Dash web UI) and as
the pure-computation Rust crate `crates/moire-core/` (driving the egui desktop app) — and
both implementations are covered on every page. The reference paper for the core moire /
CPDM physics is [Nature 652, 335 (2026); arXiv:2602.22637](../references.md#ref-wang-2026).

## Page index

| Theory page | Python module | Rust module | Status |
|---|---|---|---|
| [Moire patterns](moire-patterns.md) | `src/waytogocoop/computation/moire.py` | `crates/moire-core/src/moire.rs` | Validated — plane-wave superposition |
| [Gap modulation](gap-modulation.md) | `src/waytogocoop/computation/superconducting.py` | `crates/moire-core/src/density.rs` | Validated — BCS proximity with moire amplitude scaling |
| [Fourier analysis](fourier-analysis.md) | `src/waytogocoop/computation/fourier.py` | `crates/moire-core/src/fft.rs` | Validated — peak detection over power spectrum |
| [Isotope effects](isotope-effects.md) | `src/waytogocoop/computation/isotope_effects.py` | `crates/moire-core/src/isotope_effects.rs` | **SPECULATIVE** — no direct Te-isotope data for FeTe; alpha taken from Ba(Fe,Co)2As2 consensus |
| [Topological proximity](topological.md) | `src/waytogocoop/computation/topological.py` | `crates/moire-core/src/topological.rs` | **SPECULATIVE** — Majorana modes; 3D extensions of the 2D model |
| [Magnetic field](magnetic.md) | `src/waytogocoop/computation/magnetic.py` | `crates/moire-core/src/magnetic.rs` | **SPECULATIVE** — Abrikosov vortex lattice + Zeeman / Pauli limits; simplified models not validated for these heterostructures |
| [Graphene stacks](graphene-stacks.md) | `src/waytogocoop/computation/graphene.py` | `crates/moire-core/src/graphene.rs` | Validated, except the flat-band SC dome (**SPECULATIVE**) |
| [Bistritzer-MacDonald model](bm-model.md) | `src/waytogocoop/computation/bm_model.py` | `crates/moire-core/src/bm_model.rs` | Validated¹ |
| [Curvature](curvature.md) | `src/waytogocoop/computation/curvature.py` | `crates/moire-core/src/curvature.rs` | Validated strain / pseudo-field; gap suppression **SPECULATIVE** |

¹ The **SPECULATIVE** flat-band superconducting dome plotted alongside BM results lives in the
graphene-stacks module (`compute_flat_band_sc` in `src/waytogocoop/computation/graphene.py`),
not in `bm_model.py`; the BM band structure and DOS themselves are established physics with a
chiral-limit particle-hole-symmetry test anchor in both languages.

Isotope mass/abundance data backing the isotope-effects page comes from
`src/waytogocoop/materials/isotopes.py` and `crates/moire-core/src/isotopes.rs`.

## How to read the status column

The statuses mirror the in-app "About / Physics Reference" accordion
(`src/waytogocoop/pages/home.py`, and its desktop counterpart):

- **Validated** — established textbook physics (plane-wave moire superposition, BCS-style
  gap modulation, FFT peak detection, moire band structure), cross-checked between the
  Python and Rust implementations by mirrored test suites.
- **SPECULATIVE** — simplified models not validated for these specific heterostructures.
  Their outputs carry a `(SPECULATIVE)` tag in figure titles throughout both apps. Treat
  them as exploratory illustrations, not quantitative predictions.

## Shared page template

Every theory page follows the same structure, so you can jump straight to the section you need:

1. **Overview** — what is modeled and which app pages display it.
2. **Model** — the equations actually implemented, with derivations where the README stops short.
3. **Assumptions & validity** — where the model applies and where it breaks down.
4. **Speculative components** — present only on pages with SPECULATIVE content; mirrors the
   in-app warning language.
5. **Implementation** — a concept-to-code table (concept | Python file | Rust file) with
   function-level pointers.
6. **References** — deep links into the [annotated bibliography](../references.md).

## Suggested reading order

Start with [moire patterns](moire-patterns.md) → [gap modulation](gap-modulation.md) →
[Fourier analysis](fourier-analysis.md), which form the validated core pipeline (pattern →
gap field → spectrum). The [graphene stacks](graphene-stacks.md) →
[BM model](bm-model.md) → [curvature](curvature.md) sequence is self-contained and can be
read independently. The three SPECULATIVE pages ([isotope effects](isotope-effects.md),
[topological proximity](topological.md), [magnetic field](magnetic.md)) each build on the
validated core and are best read after it.
