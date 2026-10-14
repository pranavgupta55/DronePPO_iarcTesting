"""The run menu overlay (Tab in the viewer): tabs for Groups / Tree / Newest / Best (any stat) / Favorites, a list with a detail panel, and histograms of the shown runs.
Keys: Up/Down move (hold to repeat, Shift moves 10, the list wraps around), PageUp/PageDown, Home/End, Left/Right switch tab (also 1-5), Enter flies the run (or folds a header),
Space folds a group or branch, P pins a favorite, S cycles the Best stat, R reverses its order, T cycles a tag filter (Backspace clears it), A shows archived runs,
H cycles the histogram range (10/20/50/100/all runs of the current tab, Shift+H goes back), G switches the histogram stat page, Tab/Esc close."""
import time

import pygame

from drone_rl.viewer.fontDict import fonts
from drone_rl.viewer.menu_model import (MenuModel, TABS, STATS, STAT_KEYS, HIST_PAGES, histogram, value, fmt_date, fmt_stat)
from drone_rl import registry

WHITE, GREY, DIM, ACCENT, PINK, BG, ROW_SEL = (255, 255, 255), (150, 150, 150), (95, 98, 110), (237, 171, 80), (255, 96, 141), (24, 27, 36), (52, 58, 82)
BAR = (137, 133, 181)
ROW_H, MARGIN, DETAIL_H = 24, 56, 150
HIST_MIN_PANEL = 1250  # panels narrower than this have no room for the histograms
HIST_W = 520
BASE_COLUMNS = ["reaches_per_min", "crash_rate", "leg_s_median", "mean_speed", "train.epochs"]  # always shown in the flat tabs; the Best stat is added if it isn't one of these
NAV_KEYS = (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_PAGEUP, pygame.K_PAGEDOWN)
REPEAT_DELAY, REPEAT_LIST, REPEAT_TABS = 0.35, 0.035, 0.16  # seconds before a held key repeats, and the interval for list moves and tab switches


class MenuView:
    def __init__(self, rows, current_id):
        self.model = MenuModel(rows, registry.favorites(), current_id)
        self.model.focus(current_id)
        self.message = ""
        self.visible = 10
        self.held = {}  # nav key -> time of its next repeat

    # ---- keys
    def close(self):
        self.held.clear()

    def update(self):
        """Call once per frame while the menu is open: repeats held navigation keys."""
        now = time.time()
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        for key, due in list(self.held.items()):
            if now >= due:
                self._navigate(key, shift)
                interval = REPEAT_TABS if key in (pygame.K_LEFT, pygame.K_RIGHT) else REPEAT_LIST
                self.held[key] = now + interval

    def _move(self, delta):
        self.model.move(delta)

    def _navigate(self, key, shift):
        m = self.model
        if key == pygame.K_UP:
            self._move(-10 if shift else -1)
        elif key == pygame.K_DOWN:
            self._move(10 if shift else 1)
        elif key == pygame.K_PAGEUP:
            self._move(-self.visible)
        elif key == pygame.K_PAGEDOWN:
            self._move(self.visible)
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            m.set_tab(m.tab + (1 if key == pygame.K_RIGHT else -1))
            m.focus(m.current_id)

    def handle(self, event):
        """Returns ("load", run_id), ("close",) or None."""
        m = self.model
        if event.type == pygame.KEYUP:
            self.held.pop(event.key, None)
            return None
        if event.type == pygame.MOUSEWHEEL:
            m.cursor = max(0, m.cursor - event.y * 3)
            return None
        if event.type != pygame.KEYDOWN:
            return None
        key = event.key
        shift = bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT)
        self.message = ""
        if key in (pygame.K_TAB, pygame.K_ESCAPE):
            self.close()
            return ("close",)
        if key in NAV_KEYS:
            self._navigate(key, shift)
            self.held[key] = time.time() + REPEAT_DELAY
        elif key == pygame.K_HOME:
            m.cursor = 0
        elif key == pygame.K_END:
            m.cursor = 10 ** 6
        elif key == pygame.K_SPACE:
            e = m.selected()
            if e and (e["kind"] == "header" or e.get("kids")):
                m.collapsed.symmetric_difference_update({e["id"]})
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
        elif key == pygame.K_h:
            m.cycle_hist_range(-1 if shift else 1)
        elif key == pygame.K_g:
            m.cycle_hist_page()
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
                    self.close()
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
        show_hist = w >= HIST_MIN_PANEL
        list_right = x0 + w - (HIST_W + 24 if show_hist else 12)
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
        cols = list(BASE_COLUMNS)
        if TABS[m.tab] == "Best" and STAT_KEYS[m.stat] not in cols:
            cols.append(STAT_KEYS[m.stat])
        widths = {k: wd for k, _, wd, _, _ in STATS}
        heads = {k: hd for k, hd, _, _, _ in STATS}
        name_w, date_w = 300, 150
        col_x, cx = {}, x0 + 16 + name_w + date_w
        for k in cols:
            col_x[k] = cx
            cx += widths[k] * 8 + 12
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
                pygame.draw.rect(surface, ROW_SEL, (x0 + 6, y - 2, list_right - x0 - 6, ROW_H), border_radius=3)
            indent = 16 + 16 * e["depth"]
            if e["kind"] == "header":
                folded = e["id"] in m.collapsed
                surface.blit(f.render(self._fit(f, ("+ " if folded else "- ") + e["label"], name_w + date_w - indent), True, ACCENT), (x0 + indent - 12, y))
                best = max((r.get("reaches_per_min") or 0 for r in e["members"]), default=0)
                surface.blit(f_small.render(f"best {best:.1f}/min", True, DIM), (col_x[cols[0]], y + 2))
                continue
            row = e["row"]
            archived = row.get("archived")
            color = DIM if archived else WHITE
            mark = ("> " if e["id"] == m.current_id else "") + ("* " if e["id"] in m.favorites else "")
            label = f"{mark}{e['label']}" + (f"  [{e['kids']}]" if e.get("kids") and e["id"] in m.collapsed else "")
            surface.blit(f.render(self._fit(f, label, name_w - indent + 8), True, PINK if e["id"] == m.current_id else color), (x0 + indent, y))
            surface.blit(f_small.render(fmt_date(row), True, GREY), (x0 + 16 + name_w, y + 2))
            for k in cols:
                text = fmt_stat(row, k)
                if archived and k == "train.epochs":
                    text = "-" if row.get("train.epochs") is None else f"{row['train.epochs']:.0f}"
                surface.blit(f_small.render(text, True, GREY if archived else (ACCENT if k == STAT_KEYS[m.stat] and TABS[m.tab] == "Best" else WHITE)), (col_x[k], y + 2))
        if not entries:
            surface.blit(f.render("nothing here (pin runs with P, or clear the tag filter with Backspace)", True, DIM), (x0 + 16, list_top))
        selected = m.selected()
        if show_hist:
            self._draw_histograms(surface, list_right + 12, y0 + 52, HIST_W, y0 + h - DETAIL_H - 40, selected, f, f_small)
        self._draw_detail(surface, selected, x0, y0 + h - DETAIL_H - 28, w, f_big, f, f_small)
        help_text = "Up/Down move (Shift 10, hold)   Left/Right tabs   Enter fly   Space fold   P pin   S/R sort   T tag   A archive   H range   G stats   Tab/Esc close"
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

    def _draw_histograms(self, surface, x, y, w, bottom, selected, f, f_small):
        """Distribution of six stats over the first N runs of the current tab (N from H). Pink line = the highlighted run, white line = the flying run."""
        m = self.model
        rows = m.hist_rows()
        surface.blit(f.render(f"stats of {m.hist_label()}", True, ACCENT), (x, y))
        hint = "H range   G stats page"
        surface.blit(f_small.render(hint, True, DIM), (x + w - f_small.size(hint)[0], y + 3))
        keys = HIST_PAGES[m.hist_page]
        cols, rows_n, gap = 2, 3, 16
        cell_w = (w - gap) // cols
        cell_h = max(80, (bottom - y - 30) // rows_n)
        sel_row = selected["row"] if selected and selected["kind"] == "run" else None
        cur_row = m.by_id.get(m.current_id)
        meta = {k: (hd, fmt) for k, hd, _, fmt, _ in STATS}
        for i, key in enumerate(keys):
            cx, cy = x + (i % cols) * (cell_w + gap), y + 30 + (i // cols) * cell_h
            title, fmt = meta.get(key, (key, "{:.2f}"))
            counts, low, high = histogram([value(r, key) for r in rows])
            surface.blit(f_small.render(title, True, WHITE), (cx, cy))
            plot_top, plot_h = cy + 20, cell_h - 56
            pygame.draw.line(surface, DIM, (cx, plot_top + plot_h), (cx + cell_w, plot_top + plot_h), 1)
            if counts is None:
                surface.blit(f_small.render("no data", True, DIM), (cx + 4, plot_top + plot_h // 2 - 8))
                continue
            biggest = max(counts)
            bw = cell_w / len(counts)
            for j, c in enumerate(counts):
                if c:
                    bh = max(2, int(plot_h * c / biggest))
                    pygame.draw.rect(surface, BAR, (cx + j * bw + 1, plot_top + plot_h - bh, max(1, bw - 2), bh))
            span = (high - low) or 1.0
            for row, color in ((cur_row, WHITE), (sel_row, PINK)):
                v = value(row, key) if row else None
                if v is not None and low <= v <= high:
                    px = cx + (v - low) / span * (cell_w - 1)
                    pygame.draw.line(surface, color, (px, plot_top), (px, plot_top + plot_h), 2 if color == PINK else 1)
            vals = sorted(v for v in (value(r, key) for r in rows) if v is not None)
            surface.blit(f_small.render(fmt.format(low), True, GREY), (cx, plot_top + plot_h + 3))
            hi_text = fmt.format(high)
            surface.blit(f_small.render(hi_text, True, GREY), (cx + cell_w - f_small.size(hi_text)[0], plot_top + plot_h + 3))
            mid = f"n {len(vals)}  median {fmt.format(vals[len(vals) // 2])}"
            surface.blit(f_small.render(mid, True, DIM), (cx + (cell_w - f_small.size(mid)[0]) // 2, plot_top + plot_h + 20))

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
        surface.blit(f_small.render("   ".join(facts)[:170], True, GREY), (x0 + 16, y + 80))
        hidden = r.get("hp.hidden")
        net = f"net {r.get('model.obs_dim', '?')}-{'-'.join(str(x) for x in hidden)}-4 ({g('model.params')} params)" if hidden else "no network"
        wall = r.get("train.wall_seconds")
        train = f"{g('train.epochs')} epochs" + (f", {wall / 60:.1f} min" if isinstance(wall, (int, float)) else "")
        lr = f"lr {r.get('hp.pi_lr')} decay to {r.get('hp.lr_decay_end')} ({r.get('hp.lr_schedule')})" if r.get("hp.pi_lr") is not None else ""
        samples = f"{(r.get('train.samples_per_epoch') or 0):,} samples/epoch" if r.get("train.samples_per_epoch") else ""
        surface.blit(f_small.render("   ".join(x for x in (net, train, samples, lr, f"seed {g('hp.seed')}") if x)[:190], True, GREY), (x0 + 16, y + 100))
        tags = " ".join(f"#{t}" for t in r.get("tags", []))
        surface.blit(f_small.render(tags[:230], True, (137, 133, 181)), (x0 + 16, y + 120))
