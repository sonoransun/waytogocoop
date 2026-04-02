"""Dash application factory and entry point."""

from __future__ import annotations

import dash
from dash import Dash, html
import dash_bootstrap_components as dbc


def create_app() -> Dash:
    """Build and return the Dash application instance."""
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
    )
    app.layout = dbc.Container(
        [
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Home", href="/")),
                    dbc.NavItem(dbc.NavLink("Moire Viewer", href="/viewer")),
                    dbc.NavItem(dbc.NavLink("Parameter Sweep", href="/sweep")),
                    dbc.NavItem(dbc.NavLink("Fourier Analysis", href="/fourier")),
                    dbc.NavItem(dbc.NavLink("Substrate Comparison", href="/comparison")),
                ],
                brand="Good Job Coop!",
                brand_href="/",
                color="primary",
                dark=True,
            ),
            dash.page_container,
        ],
        fluid=True,
        className="px-0",
    )
    return app


def main() -> None:
    """Run the development server."""
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
