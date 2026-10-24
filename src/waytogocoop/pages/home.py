"""Home page — project overview and material database table."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from waytogocoop.materials.database import list_materials

dash.register_page(__name__, path="/", name="Home", title="Good Job Coop! - Home")


def _materials_table_data() -> list[dict]:
    """Build a list of dicts suitable for a Dash DataTable."""
    rows = []
    for m in list_materials():
        rows.append(
            {
                "Formula": m.formula,
                "Name": m.name,
                "Lattice": m.lattice_type,
                "a (A)": m.a,
                "c (A)": m.c,
                "Space Group": m.space_group,
                "Role": m.role,
            }
        )
    return rows


layout = dbc.Container(
    [
        html.Br(),
        dbc.Row(
            dbc.Col(
                [
                    html.H1("Moire Cooper-Pair Density Modulation"),
                    html.Hr(),
                    html.P(
                        "This tool visualises moire superlattice patterns formed when "
                        "topological insulator layers (Sb2Te3, Bi2Te3) are stacked on "
                        "FeTe substrates, and the resulting Cooper-pair density "
                        "modulation (CPDM) in the superconducting gap.",
                        className="lead",
                    ),
                    html.P(
                        "The moire interference between two lattices with different "
                        "periodicities generates a long-wavelength superlattice "
                        "potential that spatially modulates the proximity-induced "
                        "superconducting gap.  This has been observed experimentally "
                        "in topological insulator / iron-chalcogenide heterostructures."
                    ),
                ],
                width=12,
            )
        ),
        html.Br(),
        dbc.Row(
            dbc.Col(
                [
                    html.H3("Material Database"),
                    dash_table.DataTable(
                        id="home-materials-table",
                        columns=[
                            {"name": c, "id": c}
                            for c in [
                                "Formula",
                                "Name",
                                "Lattice",
                                "a (A)",
                                "c (A)",
                                "Space Group",
                                "Role",
                            ]
                        ],
                        data=_materials_table_data(),
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "8px"},
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#2c3e50",
                            "color": "white",
                        },
                    ),
                ],
                width=12,
            )
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Moire Viewer"),
                                html.P(
                                    "Interactive real-space moire pattern and gap "
                                    "modulation visualisation."
                                ),
                                dbc.Button(
                                    "Open Viewer",
                                    href="/viewer",
                                    color="primary",
                                ),
                            ]
                        )
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Parameter Sweep"),
                                html.P(
                                    "Sweep lattice mismatch or twist angle and plot "
                                    "moire period and CPDM amplitude."
                                ),
                                dbc.Button(
                                    "Open Sweep",
                                    href="/sweep",
                                    color="primary",
                                ),
                            ]
                        )
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Fourier Analysis"),
                                html.P(
                                    "FFT power spectrum of the moire pattern with "
                                    "automated peak detection."
                                ),
                                dbc.Button(
                                    "Open Fourier",
                                    href="/fourier",
                                    color="primary",
                                ),
                            ]
                        )
                    ),
                    md=4,
                ),
            ]
        ),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                html.P(
                                    [
                                        "This application visualizes moire superlattice "
                                        "patterns and the Cooper-pair density modulation "
                                        "(CPDM) they induce in the proximity-coupled "
                                        "superconducting gap. For the physics, see the "
                                        "preprint: ",
                                        html.A(
                                            "arXiv:2602.22637",
                                            href="https://arxiv.org/abs/2602.22637",
                                            target="_blank",
                                        ),
                                        ".",
                                    ]
                                ),
                                html.H6("Validated modules"),
                                html.Ul(
                                    [
                                        html.Li("Moire patterns — plane-wave superposition."),
                                        html.Li(
                                            "Gap modulation — BCS proximity with moire "
                                            "amplitude scaling."
                                        ),
                                        html.Li(
                                            "FFT analysis — peak detection over power spectrum."
                                        ),
                                    ]
                                ),
                                html.H6("Speculative modules"),
                                html.Ul(
                                    [
                                        html.Li(
                                            [
                                                "Isotope effects on the gap and coherence length "
                                                "(no direct Te-isotope data for FeTe; α = 0.4 is "
                                                "the corrected iron-based isotope-effect "
                                                "consensus, ",
                                                html.A(
                                                    "Khasanov et al., PRB 82, 212505 (2010)",
                                                    href="https://doi.org/10.1103/PhysRevB.82.212505",
                                                    target="_blank",
                                                ),
                                                "). Includes a HIGHLY SPECULATIVE "
                                                "exotic/synthetic-isotope mode (radioactive and "
                                                "hypothetical masses — what-if illustrations "
                                                "only).",
                                            ]
                                        ),
                                        html.Li(
                                            "Topological proximity / Majorana modes — the 3D "
                                            "proximity z-decay (z-slices, decay profile) and "
                                            "vortex-bound Majorana modes, 3D extensions of the "
                                            "2D model."
                                        ),
                                        html.Li(
                                            "Abrikosov vortex lattice + Zeeman / Pauli limits — "
                                            "simplified models not validated for these "
                                            "heterostructures."
                                        ),
                                        html.Li(
                                            [
                                                "Spin-fluctuation pairing — Boltzmann "
                                                "magnetotransport of electron-doped LCCO in "
                                                "which the anisotropic T-linear scattering "
                                                "tracks Tc (",
                                                html.A(
                                                    "Duffy et al., Nat. Phys. 2026",
                                                    href="https://doi.org/10.1038/s41567-026-03454-6",
                                                    target="_blank",
                                                ),
                                                "), plus an empirical pairing map and a moire "
                                                "glue-modulation what-if.",
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                "Pair-density-wave modalities — Landau model of "
                                                "the PDW, its composite charge orders and their "
                                                "survival above Tc in UTe2 (",
                                                html.A(
                                                    "Zhu et al., PNAS 2026",
                                                    href="https://doi.org/10.1073/pnas.2602117123",
                                                    target="_blank",
                                                ),
                                                "), plus a moire-locked what-if for these "
                                                "heterostructures.",
                                            ]
                                        ),
                                    ]
                                ),
                                html.P(
                                    "Speculative module outputs carry a (SPECULATIVE) tag in "
                                    "the title. Treat them as exploratory illustrations, not "
                                    "quantitative predictions.",
                                    className="text-muted small",
                                ),
                            ],
                            title="About / Physics Reference",
                        )
                    ],
                    start_collapsed=True,
                ),
                width=12,
            )
        ),
        html.Br(),
    ],
    fluid=True,
    className="p-4",
)
