from pathlib import Path

import pygame
pygame.init()

FONT_DIR = Path(__file__).parent / "fonts"

styles = ["regular", "bold", "thin", "extralight"]
sizes = [i for i in range(100)]

fonts = {
    f"{style}{size}": pygame.font.Font(str(FONT_DIR / f"Montserrat-{style.capitalize()}.ttf"), size) for style in styles for size in sizes}
