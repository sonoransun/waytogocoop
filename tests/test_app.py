"""Tests for the application shell: ``create_app()``, its navbar and theme toggle.

The navbar is one row: Home, the Core menu, Graphene, the Speculative menu and
the theme toggle. ``NAVBAR`` in ``waytogocoop.app`` is the table the user guide
quotes, and every registered page must be reachable from it.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import pytest
from dash import dcc

from waytogocoop.app import NAV_LOCATION_ID, NAVBAR, create_app, nav_item_id, nav_menu_id

# Importing the pages builds the home page's DataTable, which makes Dash 4 announce
# the component's future removal. Unrelated to these tests.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:(?s).*dash_table\.DataTable will be removed:DeprecationWarning"
)

CORE = (
    ("Moire Viewer", "/viewer"),
    ("Parameter Sweep", "/sweep"),
    ("Fourier Analysis", "/fourier"),
    ("Substrate Comparison", "/comparison"),
)
SPECULATIVE = (
    ("Magnetic Field", "/magnetic"),
    ("3D Proximity", "/proximity3d"),
    ("Phase Diagram", "/phase"),
    ("Spin Fluctuations", "/spinfluct"),
    ("Pair Density Waves", "/pdw"),
)


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None and not isinstance(children, (str, int, float)):
        yield from _walk(children)


def _links(tree) -> list[str]:
    """hrefs of the navigation links in ``tree`` (plain links and menu items), in order."""
    return [
        node.href
        for node in _walk(tree)
        if isinstance(node, (dbc.NavLink, dbc.DropdownMenuItem)) and getattr(node, "href", None)
    ]


def _table_hrefs() -> list[str]:
    hrefs: list[str] = []
    for _label, target in NAVBAR:
        hrefs += [target] if isinstance(target, str) else [href for _name, href in target]
    return hrefs


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(scope="module")
def navbar(app) -> dbc.NavbarSimple:
    return next(node for node in _walk(app.layout) if isinstance(node, dbc.NavbarSimple))


class TestNavbar:
    def test_table_is_the_documented_one(self):
        assert NAVBAR == (
            ("Home", "/"),
            ("Core", CORE),
            ("Graphene", "/graphene"),
            ("Speculative", SPECULATIVE),
        )

    def test_every_registered_page_has_exactly_one_link(self, app):
        paths = sorted(page["path"] for page in dash.page_registry.values())
        assert len(paths) == 11
        assert sorted(_links(app.layout)) == paths
        assert sorted(_table_hrefs()) == paths

    def test_labels_are_the_registered_page_names(self, app):
        names = {page["path"]: page["name"] for page in dash.page_registry.values()}
        for label, target in NAVBAR:
            if isinstance(target, str):
                assert names[target] == label
            else:
                assert [names[href] for _name, href in target] == [name for name, _ in target]

    def test_one_row_of_home_two_menus_graphene_and_the_toggle(self, navbar):
        home, core, graphene, speculative, toggle = navbar.children
        assert (home.children.children, home.children.href) == ("Home", "/")
        assert home.children.active == "exact"
        assert (graphene.children.children, graphene.children.href) == ("Graphene", "/graphene")
        assert graphene.children.active == "exact"
        for menu, (label, items) in zip((core, speculative), (NAVBAR[1], NAVBAR[3]), strict=True):
            assert isinstance(menu, dbc.DropdownMenu)
            assert (menu.label, menu.id) == (label, nav_menu_id(label))
            assert menu.nav and menu.in_navbar
            assert [(item.children, item.href, item.id) for item in menu.children] == [
                (name, href, nav_item_id(href)) for name, href in items
            ]
        assert toggle.children.id == "theme-toggle-btn"
        assert (navbar.brand, navbar.brand_href) == ("Good Job Coop!", "/")

    def test_menu_ids(self):
        assert nav_menu_id("Speculative") == "nav-menu-speculative"
        assert nav_item_id("/proximity3d") == "nav-item-proximity3d"

    def test_shell_adds_no_dcc_dropdown_or_radioitems(self, navbar):
        assert not any(isinstance(node, (dcc.Dropdown, dcc.RadioItems)) for node in _walk(navbar))

    def test_highlight_callback_follows_the_url(self, app):
        """The clientside callback reads the Location and writes every menu and item."""
        from dash import _callback

        location = next(node for node in _walk(app.layout) if isinstance(node, dcc.Location))
        assert (location.id, location.refresh) == (NAV_LOCATION_ID, False)
        callbacks = _callback.GLOBAL_CALLBACK_MAP
        key = next(k for k in callbacks if "nav-menu-core.toggle_class_name" in k)
        entry = callbacks[key]
        assert [i["id"] for i in entry["inputs"]] == [NAV_LOCATION_ID]
        assert "nav-menu-speculative.toggle_class_name" in key
        for href in [href for _n, href in CORE + SPECULATIVE]:
            assert f"{nav_item_id(href)}.active" in key


class TestTheme:
    def test_store_defaults_to_dark_and_persists_locally(self, app):
        store = next(node for node in _walk(app.layout) if isinstance(node, dcc.Store))
        assert (store.id, store.data, store.storage_type) == ("theme-store", "dark", "local")

    def test_stylesheet_is_darkly(self, app):
        assert any("darkly" in url.lower() for url in app.config.external_stylesheets)
