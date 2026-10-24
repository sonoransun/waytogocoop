"""Dash application factory and entry point."""

from __future__ import annotations

import json
from string import Template

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, clientside_callback, dcc

# Navbar, in display order: (label, href) is a plain link, (label, items) a
# dropdown menu of (label, href) pages. Home and Graphene stay plain links;
# the other pages are grouped by track so the bar is one row from 1280 px up
# and every page is at most two clicks away. The user guide quotes this table.
NAVBAR: tuple[tuple[str, str | tuple[tuple[str, str], ...]], ...] = (
    ("Home", "/"),
    (
        "Core",
        (
            ("Moire Viewer", "/viewer"),
            ("Parameter Sweep", "/sweep"),
            ("Fourier Analysis", "/fourier"),
            ("Substrate Comparison", "/comparison"),
        ),
    ),
    ("Graphene", "/graphene"),
    (
        "Speculative",
        (
            ("Magnetic Field", "/magnetic"),
            ("3D Proximity", "/proximity3d"),
            ("Phase Diagram", "/phase"),
            ("Spin Fluctuations", "/spinfluct"),
            ("Pair Density Waves", "/pdw"),
        ),
    ),
)
NAV_LOCATION_ID = "app-location"


def nav_menu_id(label: str) -> str:
    """Component id of a navbar dropdown menu, e.g. "nav-menu-core"."""
    return f"nav-menu-{label.lower()}"


def nav_item_id(href: str) -> str:
    """Component id of a page's menu item, e.g. "nav-item-viewer" for "/viewer"."""
    return f"nav-item-{href.strip('/')}"


def _navbar_children() -> list:
    children: list = []
    for label, target in NAVBAR:
        if isinstance(target, str):
            # active="exact" highlights the link on its own page.
            children.append(dbc.NavItem(dbc.NavLink(label, href=target, active="exact")))
        else:
            children.append(
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem(name, href=href, id=nav_item_id(href))
                        for name, href in target
                    ],
                    label=label,
                    id=nav_menu_id(label),
                    nav=True,
                    in_navbar=True,
                )
            )
    children.append(
        dbc.NavItem(
            dbc.Button(
                "Light Mode",
                id="theme-toggle-btn",
                color="outline-light",
                size="sm",
                className="ms-3",
            )
        )
    )
    return children


# Marks the current page's menu (toggle class "active") and item (active=True)
# from the URL path; the menus' own links give no such hint.
_HIGHLIGHT_JS = Template(
    """
    function(pathname) {
        const menus = $menus;
        const hrefs = $hrefs;
        const path = (pathname || "/").replace(/\\/+$$/, "") || "/";
        const toggles = menus.map(m => m.includes(path) ? "active" : "");
        return toggles.concat(hrefs.map(h => h === path));
    }
    """
)


def _register_navbar_highlight() -> None:
    menus = [
        (label, [href for _name, href in items])
        for label, items in NAVBAR
        if not isinstance(items, str)
    ]
    hrefs = [href for _label, items in menus for href in items]
    clientside_callback(
        _HIGHLIGHT_JS.substitute(
            menus=json.dumps([items for _label, items in menus]), hrefs=json.dumps(hrefs)
        ),
        *[Output(nav_menu_id(label), "toggle_class_name") for label, _items in menus],
        *[Output(nav_item_id(href), "active") for href in hrefs],
        Input(NAV_LOCATION_ID, "pathname"),
    )


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
            dcc.Location(id=NAV_LOCATION_ID, refresh=False),
            dbc.NavbarSimple(
                children=_navbar_children(),
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

    _register_navbar_highlight()

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
