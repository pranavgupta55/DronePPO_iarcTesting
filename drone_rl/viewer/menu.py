"""The run menu overlay (Tab in the viewer): tabs for Groups / Tree / Newest / Best (any stat) / Favorites, arrow-key scrolling, a detail panel for the highlighted run.
Keys: Up/Down/PageUp/PageDown/Home/End move, Enter flies the run (or folds a header), Left/Right fold/unfold, 1-5 pick a tab, P pins a favorite, S cycles the Best stat,
R reverses its order, T cycles a tag filter (Backspace clears it), A shows archived runs, Tab/Esc close."""
import pygame

from drone_rl.viewer.fontDict import fonts
from drone_rl.viewer.menu_model import MenuModel, TABS, STATS, STAT_KEYS, fmt_date, fmt_stat
from drone_rl import registry

WHITE, GREY, DIM, ACCENT, PINK, BG, ROW_SEL = (255, 255, 255), (150, 150, 150), (95, 98, 110), (237, 171, 80), (255, 96, 141), (24, 27, 36), (52, 58, 82)
ROW_H, MARGIN, DETAIL_H = 24, 56, 150
BASE_COLUMNS = ["reaches_per_min", "crash_rate", "leg_s_median", "mean_speed", "train.epochs"]  # always shown in the flat tabs; the Best stat is added if it isn't one of these


class MenuView:
    def __init__(self, rows, current_id):
        self.model = MenuModel(rows, registry.favorites(), current_id)
        self.model.focus(current_id)
        self.message = ""
        self.visible = 10

    # ---- keys
    def handle(self, event):
        """Returns ("load", run_id), ("close",) or None."""
        m = self.model
        if event.type == pygame.MOUSEWHEEL:
            m.cursor -= event.y * 3
            return None
        if event.type != pygame.KEYDOWN:
            return None
        key = event.key
        self.message = ""
        if key in (pygame.K_TAB, pygame.K_ESCAPE):
            return ("close",)
        if key == pygame.K_UP:
            m.cursor -= 1
        elif key == pygame.K_DOWN:
            m.cursor += 1
        elif key == pygame.K_PAGEUP:
            m.cursor -= self.visible
        elif key == pygame.K_PAGEDOWN:
            m.cursor += self.visible
        elif key == pygame.K_HOME:
            m.cursor = 0
        elif key == pygame.K_END:
            m.cursor = 10 ** 6
        elif key == pygame.K_LEFT:
            m.toggle_collapse(True)
        elif key == pygame.K_RIGHT:
            m.toggle_collapse(False)
        elif pygame.K_1 <= key <= pygame.K_5:
            m.set_tab(key - pygame.K_1)
            m.focus(m.current_id)
        elif key == pygame.K_s:
            m.cycle_stat()
            m.set_tab(TABS.index("Best"))
        elif key == pygame.K_r:
            m.descending = not m.descending
        elif key == pygame.K_t:
            m.cycle_tag()
        elif key == pygame.K_BACKSPACE:
            m.tag = None
        elif key == pygame.K_a:
            m.show_archived = not m.show_archived
            m.cursor = 0
        elif key == pygame.K_p:
            e = m.selected()
            if e and e["kind"] == "run":
                m.toggle_favorite(e["id"])
                registry.toggle_favorite(e["id"])
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            e = m.selected()
            if e and e["kind"] == "header":
                m.collapsed.symmetric_difference_update({e["id"]})
            elif e:
                row = e["row"]
                if row.get("archived") or not row.get("flyable"):
                    self.message = "read-only: this run uses the old observation layout and can't be flown"
                else:
                    return ("load", e["id"])
        return None

    # ---- drawing
    def draw(self, surface):
        m = self.model
        W, H = surface.get_size()
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((8, 9, 14, 215))
        surface.blit(shade, (0, 0))
        x0, y0, w, h = MARGIN, 40, W - 2 * MARGIN, H - 80
        pygame.draw.rect(surface, BG, (x0, y0, w, h), border_radius=6)
        pygame.draw.rect(surface, DIM, (x0, y0, w, h), 1, border_radius=6)
        f_big, f, f_small = fonts["bold20"], fonts["regular16"], fonts["regular14"]
        # tabs
        x = x0 + 16
        for i, name in enumerate(TABS):
            label = f"{i + 1} {name}"
            color = ACCENT if i == m.tab else GREY
            surface.blit(f_big.render(label, True, color), (x, y0 + 12))
            if i == m.tab:
                pygame.draw.line(surface, ACCENT, (x, y0 + 40), (x + f_big.size(label)[0], y0 + 40), 2)
            x += f_big.size(label)[0] + 28
        status = []
        if m.tag:
            status.append(f"tag: {m.tag}")
        if TABS[m.tab] == "Best":
            status.append(f"sorted by {STATS[m.stat][1]} {'high to low' if m.descending else 'low to high'}")
        if m.show_archived:
            status.append("archive shown")
        right = "   ".join(status)
        surface.blit(f.render(right, True, PINK), (x0 + w - 16 - f.size(right)[0], y0 + 16))
        # columns
        cols = [k for k in BASE_COLUMNS]
        if TABS[m.tab] == "Best" and STAT_KEYS[m.stat] not in cols:
            cols.append(STAT_KEYS[m.stat])
        widths = {k: wd for k, _, wd, _, _ in STATS}
        heads = {k: hd for k, hd, _, _, _ in STATS}
        name_w, date_w = 330, 110
        col_x, cx = {}, x0 + 16 + name_w + date_w
        for k in cols:
            col_x[k] = cx
            cx += widths[k] * 9 + 14
        header_y = y0 + 52
        surface.blit(f_small.render("run", True, DIM), (x0 + 16, header_y))
        surface.blit(f_small.render("date", True, DIM), (x0 + 16 + name_w, header_y))
        for k in cols:
            color = ACCENT if TABS[m.tab] == "Best" and k == STAT_KEYS[m.stat] else DIM
            surface.blit(f_small.render(heads[k], True, color), (col_x[k], header_y))
        # rows
        list_top = header_y + 24
        self.visible = max(3, (y0 + h - DETAIL_H - 30 - list_top) // ROW_H)
        entries = m.entries()
        m.clamp(self.visible)
        for i in range(m.top, min(len(entries), m.top + self.visible)):
            e = entries[i]
            y = list_top + (i - m.top) * ROW_H
            if i == m.cursor:
                pygame.draw.rect(surface, ROW_SEL, (x0 + 6, y - 2, w - 12, ROW_H), border_radius=3)
            indent = 16 + 16 * e["depth"]
            if e["kind"] == "header":
                folded = e["id"] in m.collapsed
                surface.blit(f.render(("+ " if folded else "- ") + e["label"], True, ACCENT), (x0 + indent - 12, y))
                best = max((r.get("reaches_per_min") or 0 for r in e["members"]), default=0)
                surface.blit(f_small.render(f"best {best:.1f} reach/min", True, DIM), (x0 + 16 + name_w, y + 2))
                continue
            row = e["row"]
            archived = row.get("archived")
            color = DIM if archived else WHITE
            mark = ("> " if e["id"] == m.current_id else "") + ("* " if e["id"] in m.favorites else "")
            label = f"{mark}{e['label']}" + (f"  [{e['kids']}]" if e.get("kids") and e["id"] in m.collapsed else "")
            surface.blit(f.render(self._fit(f, label, name_w - indent - 12), True, PINK if e["id"] == m.current_id else color), (x0 + indent, y))
            surface.blit(f_small.render(fmt_date(row), True, GREY), (x0 + 16 + name_w, y + 2))
            for k in cols:
                text = fmt_stat(row, k)
                if archived and k == "train.epochs":
                    text = "-" if row.get("train.epochs") is None else f"{row['train.epochs']:.0f}"
                surface.blit(f_small.render(text, True, GREY if archived else (ACCENT if k == STAT_KEYS[m.stat] and TABS[m.tab] == "Best" else WHITE)), (col_x[k], y + 2))
        if not entries:
            surface.blit(f.render("nothing here (pin runs with P, or clear the tag filter with Backspace)", True, DIM), (x0 + 16, list_top))
        # detail panel
        self._draw_detail(surface, m.selected(), x0, y0 + h - DETAIL_H - 28, w, f_big, f, f_small)
        help_text = "Up/Down move   Enter fly   Left/Right fold   1-5 tabs   P pin   S sort stat   R reverse   T tag   A archive   Tab/Esc close"
        if self.message:
            surface.blit(f_small.render(self.message, True, PINK), (x0 + 16, y0 + h - 24))
        else:
            surface.blit(f_small.render(help_text, True, DIM), (x0 + 16, y0 + h - 24))

    @staticmethod
    def _fit(font, text, width):
        """text shortened with an ellipsis to fit width px"""
        if font.size(text)[0] <= width:
            return text
        while len(text) > 1 and font.size(text + "...")[0] > width:
            text = text[:-1]
        return text + "..."

    def _draw_detail(self, surface, entry, x0, y, w, f_big, f, f_small):
        pygame.draw.line(surface, DIM, (x0 + 10, y), (x0 + w - 10, y), 1)
        if not entry or entry["kind"] != "run":
            return
        r = entry["row"]
        title = f"{r.get('name')}: {r.get('title')}" if r.get("title") and r.get("title") != r.get("name") else str(r.get("name"))
        surface.blit(f_big.render(title[:90], True, WHITE), (x0 + 16, y + 8))
        words, line, lines = (r.get("blurb") or "").split(), "", []
        for word in words:
            test = f"{line} {word}".strip()
            if f.size(test)[0] > w - 40:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
        for i, text in enumerate(lines[:2]):
            surface.blit(f.render(text, True, (200, 200, 200)), (x0 + 16, y + 36 + i * 20))
        g = lambda k, d="-": r.get(k) if r.get(k) is not None else d
        world = r.get("world")
        parent = self.model.by_id.get(r.get("parent"), {}).get("name") if r.get("parent") else None
        facts = [f"group: {g('group')}", f"parent: {parent or '-'}", f"kind: {g('kind')}", f"world: {int(world[0])}x{int(world[1])}" if world else "world: ?"]
        surface.blit(f_small.render("   ".join(facts)[:160], True, GREY), (x0 + 16, y + 80))
        hidden = r.get("hp.hidden")
        net = f"net {r.get('model.obs_dim', '?')}-{'-'.join(str(x) for x in hidden)}-4 ({g('model.params')} params)" if hidden else "no network"
        wall = r.get("train.wall_seconds")
        train = f"{g('train.epochs')} epochs" + (f", {wall / 60:.1f} min" if isinstance(wall, (int, float)) else "")
        lr = f"lr {r.get('hp.pi_lr')} decay to {r.get('hp.lr_decay_end')} ({r.get('hp.lr_schedule')})" if r.get("hp.pi_lr") is not None else ""
        samples = f"{(r.get('train.samples_per_epoch') or 0):,} samples/epoch" if r.get("train.samples_per_epoch") else ""
        surface.blit(f_small.render("   ".join(x for x in (net, train, samples, lr, f"seed {g('hp.seed')}") if x)[:170], True, GREY), (x0 + 16, y + 100))
        tags = " ".join(f"#{t}" for t in r.get("tags", []))
        surface.blit(f_small.render(tags[:200], True, (137, 133, 181)), (x0 + 16, y + 120))
