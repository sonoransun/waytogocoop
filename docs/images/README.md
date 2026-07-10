# docs/images

[← Documentation index](../README.md)

Every PNG in this directory is **script-generated — never hand-captured**.
Do not edit or replace images manually; rerun the matching capture script
instead. Both scripts are deterministic: two runs produce byte-identical
PNGs, so regenerated images diff cleanly.

## Regenerating

Python scenes (Plotly + kaleido; the Chromium fetch is a one-time install
step):

```bash
.venv/bin/plotly_get_chrome -y
python scripts/capture_screenshots.py
```

Rust scenes (headless CPU software rasterizer, no display needed):

```bash
cargo run -p moire-desktop --bin capture --release
```

Either script accepts `--only <scene>` to regenerate a single image. See the
"Screenshot capture" section of
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#screenshot-capture) for details.

## Image inventory

Scenes named `rust-*` are defined in `crates/moire-desktop/src/bin/capture.rs`;
all others in the `SCENES` dict of `scripts/capture_screenshots.py`.

| Filename | Generator scene | Caption |
|---|---|---|
| `viewer-2d.png` | `viewer-2d` | Moire Viewer — real-space pattern of Sb₂Te₃ / FeTe (paper default preset). |
| `viewer-3d.png` | `viewer-3d` | Moire Viewer 3D — triangulated `go.Mesh3d` surface (WebGL-friendly path for grid ≥ 150). |
| `fourier.png` | `fourier` | FFT Analysis — log₁₀(\|F\|²) power spectrum showing the moire reciprocal vectors, shared-LUT `inferno`. |
| `proximity-3d.png` | `proximity-3d` | Proximity 3D — nested Δ(x,y,z) isosurfaces annotated with the z=0 interface and ξ_prox. |
| `proximity-3d-volume.png` | `proximity-3d-volume` | Proximity 3D — true volumetric rendering via `go.Volume`. |
| `proximity-3d-clipped.png` | `proximity-3d-clipped` | Proximity 3D — clipping plane at z=100 Å reveals the interior decay profile. |
| `magnetic-currents-3d.png` | `magnetic-currents-3d` | Magnetic — screening currents as native `go.Cone` quivers on a translucent gap surface at Bz = 4 T. |
| `magnetic-vortex-2d.png` | `magnetic-vortex-2d` | Magnetic — moire gap heatmap with Abrikosov vortex cores overlaid as markers at Bz = 4 T. |
| `magnetic-majorana-3d.png` | `magnetic-majorana-3d` | Majorana ZM 3D Density (SPECULATIVE) — isosurface of the envelope × Bessel oscillation × z-decay at Bz = 4 T. |
| `phase-diagram.png` | `phase-diagram` | Fu-Kane phase boundary (SPECULATIVE) over (B, Δ) space. |
| `sweep-twist.png` | `sweep-twist` | Parameter Sweep — moire period and CPDM amplitude vs twist angle for an FeTe homo-bilayer. |
| `graphene-pattern.png` | `graphene-pattern` | Twisted-bilayer graphene moire pattern at the magic angle θ = 1.08° (L ≈ 130 Å). |
| `graphene-bands.png` | `graphene-bands` | Bistritzer-MacDonald band structure along K → Γ → M → K′ at θ = 1.08°, flat bands highlighted. |
| `graphene-dos.png` | `graphene-dos` | Gaussian-broadened BM density of states at θ = 1.08° — the flat-band peak at E = 0. |
| `graphene-pseudo-field.png` | `graphene-pseudo-field` | Pseudo-magnetic field (SPECULATIVE) of a 2 Å sinusoidal ripple oriented 30° from zigzag, K valley. |
| `graphene-curved-3d.png` | `graphene-curved-3d` | Curved sheet (SPECULATIVE) — Gaussian bump (5 Å, σ = 50 Å) colored by the flat-band gap × pseudo-field suppression. |
| `graphene-supermoire.png` | `graphene-supermoire` | Supermoire — magic-angle TBG on Sb₂Te₃ with the stack, interface, and beat periods in the title. |
| `rust-desktop-2d.png` | `rust-desktop-2d` | Rust desktop — top-down moire pattern with axes and colorbar. |
| `rust-desktop-3d.png` | `rust-desktop-3d` | Rust desktop — shaded 3D surface with world axes and scale bar. |
| `rust-desktop-wireframe.png` | `rust-desktop-wireframe` | Rust desktop — 3D surface with the wireframe toggle (W) on. |
| `rust-density-3d.png` | `rust-density-3d` | Rust desktop — Cooper-pair density modulation surface with diverging `coolwarm`. |
| `rust-graphene-pattern.png` | `rust-graphene-pattern` | Rust desktop — graphene stack pattern via `compute_graphene_stack_v2`, top view. |
| `rust-curved-3d.png` | `rust-curved-3d` | Rust desktop — curved sheet with pseudo-field coloring via `render_surface_3d_colored`. |

When the apps run live, the same figures are reachable interactively:
Plotly's camera button in the web UI and **Ctrl+S** in the desktop app
(timestamped PNG in the current working directory).
