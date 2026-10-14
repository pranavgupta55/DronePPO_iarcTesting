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
    view.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=0))
    view.draw(surface)


def test_menu_keys_wrap_shift_and_tabs():
    from drone_rl import registry
    from drone_rl.viewer.menu import MenuView
    pygame.init()
    rows = registry.load_index()
    view = MenuView(rows, rows[0]["id"])
    key = lambda k, mod=0: view.handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=""))
    key(pygame.K_3)
    n = len(view.model.entries())
    view.model.cursor = 0
    key(pygame.K_UP)
    assert view.model.cursor == n - 1  # loops from the top row to the bottom row
    view.model.cursor = 0
    key(pygame.K_DOWN, pygame.KMOD_SHIFT)
    assert view.model.cursor == 10
    tab = view.model.tab
    key(pygame.K_RIGHT)
    assert view.model.tab == tab + 1
    view.handle(pygame.event.Event(pygame.KEYUP, key=pygame.K_RIGHT, mod=0))
    assert pygame.K_RIGHT not in view.held and pygame.K_UP in view.held  # releasing a key stops only its own repeat
