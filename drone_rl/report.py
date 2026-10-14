"""Summary table and charts of the run index: python3 -m drone_rl report [--group G] [--top N]  (charts go to docs/reports/)"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from drone_rl import registry

OUT = registry.REPO / "docs" / "reports"
SHORT = ["name", "group", "kind", "tier", "score", "reaches_per_min", "crash_rate", "leg_s_median", "mean_speed", "tilt_frac", "nozzle_use_deg", "train.epochs"]


def main():
    ap = argparse.ArgumentParser(prog="drone_rl report")
    ap.add_argument("--group", default=None)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    df = pd.DataFrame(registry.load_index())
    if df.empty:
        print("no runs")
        return
    if args.group:
        df = df[df["group"] == args.group]
    ok = df[(df["status"] == "ok") & df["score"].notna()].sort_values(["tier", "score"], ascending=False)
    cols = [c for c in SHORT if c in ok.columns]
    print(f"{len(df)} runs ({args.group or 'all groups'}): {len(ok)} evaluated ok")
    if not len(ok):
        return
    print(ok[cols].head(args.top).round(3).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    best = ok.groupby("group")["score"].max().sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(best) + 1.5))
    ax.barh(best.index.astype(str), best.values)
    ax.set_xlabel("best score in group")
    fig.tight_layout()
    fig.savefig(OUT / "best_per_group.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ok["crash_rate"], ok["reaches_per_min"], s=18, alpha=0.7)
    ax.set_xlabel("crash rate (random courses)")
    ax.set_ylabel("reaches per minute")
    fig.tight_layout()
    fig.savefig(OUT / "reaches_vs_crash.png", dpi=110)
    plt.close(fig)
    print("charts in", OUT)


if __name__ == "__main__":
    main()
