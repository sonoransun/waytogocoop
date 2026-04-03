"""Dash application factory and entry point."""

from __future__ import annotations

import dash
from dash import Dash, dcc, Input, Output, clientside_callback
import dash_bootstrap_components as dbc


def create_app() -> Dash:
    """Build and return the Dash application instance."""
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )
    app.layout = dbc.Container(
        [
            dcc.Store(id="theme-store", data="dark", storage_type="local"),
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Home", href="/")),
                    dbc.NavItem(dbc.NavLink("Moire Viewer", href="/viewer")),
                    dbc.NavItem(dbc.NavLink("Parameter Sweep", href="/sweep")),
                    dbc.NavItem(dbc.NavLink("Fourier Analysis", href="/fourier")),
                    dbc.NavItem(dbc.NavLink("Substrate Comparison", href="/comparison")),
                    dbc.NavItem(dbc.NavLink("Magnetic Field", href="/magnetic")),
                    dbc.NavItem(dbc.NavLink("3D Proximity", href="/proximity3d")),
                    dbc.NavItem(dbc.NavLink("Phase Diagram", href="/phase")),
                    dbc.NavItem(
                        dbc.Button(
                            "Light Mode",
                            id="theme-toggle-btn",
                            color="outline-light",
                            size="sm",
                            className="ms-3",
                        )
                    ),
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

    # Clientside callback to swap the Bootstrap stylesheet and body class
    clientside_callback(
        """
        function(n_clicks, current_theme) {
            if (!n_clicks) {
                // Initial load: apply saved theme
                var theme = current_theme || "dark";
            } else {
                var theme = (current_theme === "dark") ? "light" : "dark";
            }

            // Swap Bootstrap stylesheet
            var links = document.querySelectorAll('link[rel="stylesheet"]');
            links.forEach(function(link) {
                var href = link.href;
                if (href.includes("DARKLY") || href.includes("darkly")) {
                    if (theme === "light") {
                        link.href = href.replace(/darkly/gi, function(m) {
                            return m === m.toUpperCase() ? "FLATLY" : "flatly";
                        });
                    }
                } else if (href.includes("FLATLY") || href.includes("flatly")) {
                    if (theme === "dark") {
                        link.href = href.replace(/flatly/gi, function(m) {
                            return m === m.toUpperCase() ? "DARKLY" : "darkly";
                        });
                    }
                }
            });

            // Set body class for CSS targeting
            document.body.classList.toggle("dark-theme", theme === "dark");
            document.body.classList.toggle("light-theme", theme === "light");

            return theme;
        }
        """,
        Output("theme-store", "data"),
        Input("theme-toggle-btn", "n_clicks"),
        Input("theme-store", "data"),
    )

    # Update button label to show opposite mode
    clientside_callback(
        """
        function(theme) {
            return (theme === "dark") ? "Light Mode" : "Dark Mode";
        }
        """,
        Output("theme-toggle-btn", "children"),
        Input("theme-store", "data"),
    )

    return app


def main() -> None:
    """Run the development server."""
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
