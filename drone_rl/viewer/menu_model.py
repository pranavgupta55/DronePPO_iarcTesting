"""The data side of the run menu (no pygame): tabs, ordering, grouping, branch tree, tag filter, favorites. menu.py draws it and handles the keys.
Rows are the flat dicts of runs/index.json (registry.index_row / archive_rows)."""
from datetime import datetime

TABS = ["Groups", "Tree", "Newest", "Best", "Favorites"]

# (key, header, width, format, higher is better); the Best tab sorts by one of these and S cycles through them
STATS = [
    ("reaches_per_min", "reach/min", 9, "{:.1f}", True),
    ("crash_rate", "crash %", 8, "{:.0%}", False),
    ("leg_s_median", "leg s", 7, "{:.2f}", False),
    ("mean_speed", "speed", 7, "{:.0f}", True),
    ("score", "score", 7, "{:.1f}", True),
    ("fixed_reached_frac", "fixed ok", 9, "{:.0%}", True),
    ("robust.noise05.reached_frac", "noisy ok", 9, "{:.0%}", True),
    ("tilt_frac", "tilt %", 7, "{:.0%}", False),
    ("nozzle_use_deg", "nozzle°", 8, "{:.1f}", True),
    ("mean_thrust", "thrust", 7, "{:.2f}", True),
    ("train.epochs", "epochs", 7, "{:.0f}", True),
    ("train.wall_seconds", "train s", 8, "{:.0f}", False),
    ("model.params", "params", 7, "{:.0f}", False),
]
STAT_KEYS = [s[0] for s in STATS]

# one line on what each stat measures (shown under the histogram titles and next to the sort options)
STAT_INFO = {
    "reaches_per_min": "waypoints reached per minute of flight, random courses",
    "crash_rate": "share of random-course flights that end by leaving the world",
    "leg_s_median": "median seconds to reach one waypoint (strict reach rule)",
    "mean_speed": "average speed while flying, px/s",
    "score": "reaches/min if all three gates pass, else scaled down",
    "fixed_reached_frac": "share of waypoints reached on the 5 fixed courses",
    "robust.noise05.reached_frac": "share of waypoints reached with action noise 0.05",
    "tilt_frac": "share of time tilted more than 15 degrees",
    "nozzle_use_deg": "average nozzle swing away from straight down, degrees",
    "mean_thrust": "average total thruster force, 0 to 2 (hover is about 0.17)",
    "train.epochs": "training epochs (PPO) the run went through",
    "train.wall_seconds": "wall-clock seconds the training took",
    "model.params": "number of weights and biases in the network",
}


def describe(key):
    return STAT_INFO.get(key, key)

# histograms: how many of the runs shown in the current tab to include (the first N in the order the tab shows them), and which stats make up each page of the panel
HIST_RANGES = [10, 20, 50, 100, None]  # None = all
HIST_PAGES = [["reaches_per_min", "crash_rate", "leg_s_median", "mean_speed", "tilt_frac", "nozzle_use_deg"],
              ["train.epochs", "train.wall_seconds", "model.params", "mean_thrust", "robust.noise05.reached_frac", "fixed_reached_frac"]]
HIST_BINS = 12


def histogram(values, bins=HIST_BINS):
    """(counts, low, high) of the numbers in values (None entries ignored); (None, None, None) when there are none."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    low, high = min(vals), max(vals)
    if high == low:
        return [len(vals)] + [0] * (bins - 1), low, low + 1e-9
    counts = [0] * bins
    for v in vals:
        counts[min(bins - 1, int((v - low) / (high - low) * bins))] += 1
    return counts, low, high


def fmt_date(row):
    """'10-13 8:20 PM' (archived runs: '~' in front, since only their file times are known)."""
    stamp = row.get("created") or row.get("finished")
    if not stamp:
        return "built-in"
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return "?"
    text = f"{when:%m-%d} {int(when.strftime('%I'))}:{when:%M %p}"
    return "~" + text if row.get("archived") else text


def fmt_stat(row, key):
    for k, _, _, fmt, _ in STATS:
        if k == key:
            value = row.get(key)
            if row.get("archived") and key == "reaches_per_min":
                value = None
            return "-" if value is None else fmt.format(value)
    return "-"


def value(row, key):
    v = row.get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class MenuModel:
    def __init__(self, rows, favorites, current_id=None):
        self.rows = rows
        self.by_id = {r["id"]: r for r in rows}
        self.favorites = list(favorites)
        self.current_id = current_id
        self.tab = 0
        self.stat = 0  # index into STAT_KEYS for the Best tab
        self.descending = True
        self.show_archived = False
        self.tag = None
        self.collapsed = set()  # group names / run ids folded away
        self.cursor = 0
        self.top = 0
        self.hist_range = 1  # index into HIST_RANGES
        self.hist_page = 0

    # ---- state changes
    def set_tab(self, index):
        self.tab = index % len(TABS)
        self.cursor = self.top = 0

    def cycle_hist_range(self, step=1):
        self.hist_range = (self.hist_range + step) % len(HIST_RANGES)

    def cycle_hist_page(self):
        self.hist_page = (self.hist_page + 1) % len(HIST_PAGES)

    def view_rows(self):
        """The distinct runs of the current tab in display order (what the histograms are computed over); archived runs only when shown."""
        seen, rows = set(), []
        for e in self.entries():
            if e["kind"] == "run" and e["id"] not in seen:
                seen.add(e["id"])
                rows.append(e["row"])
        return rows

    def hist_rows(self):
        limit = HIST_RANGES[self.hist_range]
        rows = self.view_rows()
        return rows if limit is None else rows[:limit]

    def hist_label(self):
        limit, total = HIST_RANGES[self.hist_range], len(self.view_rows())
        n = min(limit, total) if limit else total
        if limit is None:
            return f"all {total} runs"
        text = f"{n} most recent" if TABS[self.tab] == "Newest" else f"first {n} shown"
        return text + ("" if n == limit else f" (only {total} here)")

    def cycle_stat(self):
        self.stat = (self.stat + 1) % len(STATS)
        self.descending = STATS[self.stat][4]

    def cycle_tag(self):
        tags = self.available_tags()
        if not tags:
            self.tag = None
        elif self.tag not in tags:
            self.tag = tags[0]
        else:
            nxt = tags.index(self.tag) + 1
            self.tag = tags[nxt] if nxt < len(tags) else None
        self.cursor = self.top = 0

    def available_tags(self):
        count = {}
        for r in self._base_rows():
            for t in r.get("tags", []):
                count[t] = count.get(t, 0) + 1
        return sorted(count, key=lambda t: (-count[t], t))

    def toggle_favorite(self, run_id):
        self.favorites = [f for f in self.favorites if f != run_id] if run_id in self.favorites else self.favorites + [run_id]

    # ---- ordering
    def _base_rows(self):
        rows = [r for r in self.rows if self.show_archived or not r.get("archived")]
        return [r for r in rows if r.get("status") in ("ok", "archived") or r.get("archived")]

    def _filtered(self):
        rows = self._base_rows()
        return [r for r in rows if self.tag is None or self.tag in r.get("tags", [])]

    @staticmethod
    def _newest_key(r):
        return r.get("created") or r.get("finished") or ""

    def entries(self):
        """Display rows in order: {"kind": "header"|"run", "row", "depth", "label", "id"}; headers carry group stats and can be collapsed."""
        rows = self._filtered()
        tab = TABS[self.tab]
        if tab == "Newest":
            return [self._run(r, 0) for r in sorted(rows, key=self._newest_key, reverse=True)]
        if tab == "Favorites":
            favs = [self.by_id[f] for f in self.favorites if f in self.by_id and self.by_id[f] in rows]
            return [self._run(r, 0) for r in favs]
        if tab == "Best":
            key = STAT_KEYS[self.stat]
            have = [r for r in rows if value(r, key) is not None]
            sign = 1 if self.descending else -1  # ties (e.g. many runs with 0% crashes) fall back to the better reaches per minute
            return [self._run(r, 0) for r in sorted(have, key=lambda r: (-sign * value(r, key), -(value(r, "reaches_per_min") or 0)))]
        if tab == "Groups":
            groups = {}
            for r in rows:
                groups.setdefault(r.get("group") or "(no group)", []).append(r)
            out = []
            for name, members in sorted(groups.items(), key=lambda kv: max(self._newest_key(r) for r in kv[1]), reverse=True):
                out.append({"kind": "header", "id": name, "row": None, "depth": 0, "label": f"{name}  ({len(members)})", "members": members})
                if name not in self.collapsed:
                    out += [self._run(r, 1) for r in sorted(members, key=self._newest_key)]
            return out
        return self._tree_entries(rows)

    def _run(self, row, depth):
        return {"kind": "run", "id": row["id"], "row": row, "depth": depth, "label": row.get("name", row["id"])}

    def _tree_entries(self, rows):
        ids = {r["id"] for r in rows}
        children = {}
        roots = []
        for r in sorted(rows, key=self._newest_key):
            parent = r.get("parent")
            if parent in ids:
                children.setdefault(parent, []).append(r)
            else:
                roots.append(r)
        out = []

        def walk(row, depth):
            kids = children.get(row["id"], [])
            entry = self._run(row, depth)
            entry["kids"] = len(kids)
            out.append(entry)
            if row["id"] not in self.collapsed:
                for kid in kids:
                    walk(kid, depth + 1)

        for root in sorted(roots, key=self._newest_key, reverse=True):
            walk(root, 0)
        return out

    def flyable_order(self):
        """Run ids in the order shown, flyable ones only (what Left/Right step through outside the menu)."""
        seen, order = set(), []
        for e in self.entries():
            if e["kind"] == "run" and e["row"].get("flyable") and not e["row"].get("archived") and e["id"] not in seen:
                seen.add(e["id"])
                order.append(e["id"])
        return order

    # ---- cursor
    def move(self, delta, stop_at_edge=False, repeating=False):
        """Moves the cursor by delta rows, looping: past the last row comes the first and the other way round.
        With stop_at_edge (Shift+arrow, 10 rows) a move that would cross an end stops on that end's row first; a fresh press while already on it loops to the other end,
        a held key's repeat stays put."""
        n = len(self.entries())
        if not n:
            return
        target = self.cursor + delta
        if not stop_at_edge or 0 <= target < n:
            self.cursor = target % n
            return
        end = n - 1 if delta > 0 else 0
        if self.cursor != end:
            self.cursor = end
        elif not repeating:
            self.cursor = 0 if delta > 0 else n - 1

    def tag_options(self, limit=6):
        """Labels for the tag filter chips: 'all' plus the most common tags, always including the selected one. Returns (labels, selected index)."""
        tags = self.available_tags()[:limit]
        if self.tag and self.tag not in tags:
            tags = tags[:-1] + [self.tag] if tags else [self.tag]
        return ["all"] + tags, 0 if self.tag is None else 1 + tags.index(self.tag)

    def clamp(self, visible):
        n = len(self.entries())
        self.cursor = max(0, min(self.cursor, n - 1))
        if self.cursor < self.top:
            self.top = self.cursor
        if self.cursor >= self.top + visible:
            self.top = self.cursor - visible + 1
        self.top = max(0, min(self.top, max(0, n - visible)))

    def selected(self):
        entries = self.entries()
        return entries[self.cursor] if entries and 0 <= self.cursor < len(entries) else None

    def toggle_collapse(self, collapse):
        e = self.selected()
        if not e:
            return
        if e["kind"] == "header":
            (self.collapsed.add if collapse else self.collapsed.discard)(e["id"])
        elif TABS[self.tab] == "Tree" and e.get("kids"):
            (self.collapsed.add if collapse else self.collapsed.discard)(e["id"])

    def focus(self, run_id):
        for i, e in enumerate(self.entries()):
            if e["kind"] == "run" and e["id"] == run_id:
                self.cursor = i
                return True
        return False
