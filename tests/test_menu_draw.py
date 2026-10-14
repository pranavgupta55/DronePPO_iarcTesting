"""The menu draws every tab without error on a headless display (skipped where pygame isn't installed, e.g. outside the project venv)."""
import os

import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def test_menu_draws_all_tabs():
    from drone_rl import registry
    from drone_rl.viewer.menu import MenuView
    pygame.init()
    surface = pygame.display.set_mode((1024, 768))
    rows = registry.load_index()
    view = MenuView(rows, rows[0]["id"])
    for tab in range(5):
        view.model.set_tab(tab)
        view.draw(surface)
    view.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
    view.draw(surface)
