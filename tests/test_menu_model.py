"""The run menu's data model: tabs, grouping, branch tree, sorting by a stat, tag filter, favorites, the order Left/Right step through."""
from drone_rl.viewer.menu_model import MenuModel, STAT_KEYS, TABS


def row(id_, group="g", parent=None, created="2026-10-01T10:00:00", tags=("t",), reaches=5.0, flyable=True, archived=False, status="ok"):
    return {"id": id_, "name": id_, "group": group, "parent": parent, "created": created, "tags": list(tags), "reaches_per_min": reaches, "crash_rate": 0.1,
            "flyable": flyable, "archived": archived, "status": "archived" if archived else status}


ROWS = [row("a", "alpha", None, "2026-10-01T10:00:00", reaches=3.0), row("b", "alpha", "a", "2026-10-02T10:00:00", reaches=9.0),
        row("c", "beta", "b", "2026-10-03T10:00:00", ("x", "t"), reaches=6.0), row("old", "legacy", None, "2026-09-01T10:00:00", archived=True, flyable=False),
        row("bad", "beta", None, "2026-10-04T10:00:00", status="train_failed")]


def test_newest_tab_orders_by_time_and_hides_archived_and_failed():
    m = MenuModel(ROWS, [])
    m.set_tab(TABS.index("Newest"))
    assert [e["id"] for e in m.entries()] == ["c", "b", "a"]
    m.show_archived = True
    assert "old" in [e["id"] for e in m.entries()]


def test_groups_tab_has_headers_newest_group_first_and_collapses():
    m = MenuModel(ROWS, [])
    m.set_tab(TABS.index("Groups"))
    entries = m.entries()
    assert [e["kind"] for e in entries] == ["header", "run", "header", "run", "run"]
    assert entries[0]["id"] == "beta" and [e["id"] for e in entries[3:]] == ["a", "b"]
    m.collapsed.add("alpha")
    assert [e["kind"] for e in m.entries()] == ["header", "run", "header"]


def test_tree_tab_nests_children_under_parents():
    m = MenuModel(ROWS, [])
    m.set_tab(TABS.index("Tree"))
    assert [(e["id"], e["depth"]) for e in m.entries()] == [("a", 0), ("b", 1), ("c", 2)]
    m.cursor = 0
    m.toggle_collapse(True)
    assert [e["id"] for e in m.entries()] == ["a"]


def test_best_tab_sorts_by_stat_and_direction():
    m = MenuModel(ROWS, [])
    m.set_tab(TABS.index("Best"))
    assert [e["id"] for e in m.entries()] == ["b", "c", "a"]
    m.descending = False
    assert [e["id"] for e in m.entries()] == ["a", "c", "b"]
    m.cycle_stat()
    assert STAT_KEYS[m.stat] == "crash_rate"


def test_favorites_and_tag_filter_and_flyable_order():
    m = MenuModel(ROWS, ["c", "a"])
    m.set_tab(TABS.index("Favorites"))
    assert [e["id"] for e in m.entries()] == ["c", "a"]
    m.toggle_favorite("c")
    assert [e["id"] for e in m.entries()] == ["a"]
    m.set_tab(TABS.index("Newest"))
    m.tag = "x"
    assert [e["id"] for e in m.entries()] == ["c"]
    m.tag = None
    assert m.flyable_order() == ["c", "b", "a"]  # archived and failed runs are not flyable


def test_cursor_loops_both_ways_and_moves_by_ten():
    rows = [row(f"r{i}", created=f"2026-10-01T10:{i:02d}:00") for i in range(25)]
    m = MenuModel(rows, [])
    m.set_tab(TABS.index("Newest"))
    m.move(-1)
    assert m.cursor == 24  # up from the top row wraps to the bottom
    m.move(1)
    assert m.cursor == 0  # down from the bottom wraps to the top
    m.move(10)
    m.move(10)
    m.move(10)
    assert m.cursor == 5  # 30 rows down in a 25 row list


def test_dates_show_am_pm():
    from drone_rl.viewer.menu_model import fmt_date
    assert fmt_date({"created": "2026-10-13T20:05:00-04:00"}) == "10-13 8:05 PM"
    assert fmt_date({"created": "2026-10-13T00:30:00-04:00"}) == "10-13 12:30 AM"
    assert fmt_date({"finished": "2026-10-10T09:07:00-04:00", "archived": True}) == "~10-10 9:07 AM"


def test_histogram_bins_and_ranges():
    from drone_rl.viewer.menu_model import histogram, HIST_RANGES
    counts, low, high = histogram([0, 1, 1, 2, 4, None], bins=4)
    assert (low, high, sum(counts)) == (0, 4, 5) and counts == [1, 2, 1, 1]
    assert histogram([None, None]) == (None, None, None)
    assert histogram([3, 3, 3], bins=3)[0] == [3, 0, 0]
    rows = [row(f"r{i}", created=f"2026-10-{1 + i // 24:02d}T{i % 24:02d}:00:00", reaches=float(i)) for i in range(60)]
    m = MenuModel(rows, [])
    m.set_tab(TABS.index("Newest"))
    assert HIST_RANGES == [10, 20, 50, 100, None]
    seen = []
    for _ in HIST_RANGES:
        seen.append(len(m.hist_rows()))
        m.cycle_hist_range()
    assert seen == [20, 50, 60, 60, 10]  # starts on 20; 100 and all both give the 60 that exist
    m.hist_range = 0
    assert [r["id"] for r in m.hist_rows()][:2] == ["r59", "r58"]  # the most recent first
    assert m.hist_label() == "10 most recent"
    m.hist_range = 3
    assert m.hist_label() == "60 most recent (only 60 here)"
    m.hist_range = 4
    assert m.hist_label() == "all 60 runs"
    m.set_tab(TABS.index("Best"))
    m.hist_range = 1
    assert m.hist_label() == "first 20 shown"
