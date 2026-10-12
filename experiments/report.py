"""Compact summary + charts of results.csv: python3 report.py [--batch B] [--top N] [--results CSV]"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
SHORT = ["name", "scheme", "tier", "score", "reaches_per_ep", "crash_rate", "hover_tilt_frac", "tilt_streak_frac", "rock_rate", "outward_frac", "mean_abs_ang_vel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--results", default=str(HERE / "results.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    if args.batch:
        df = df[df["batch"] == args.batch]
    if df.empty:
        print("no results")
        return
    ok = df[df["status"] == "ok"].copy()
    failed = df[df["status"] != "ok"]
    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    tag = args.batch or "all"
    ok = ok.sort_values(["tier", "score"], ascending=False)

    lines = [f"{len(df)} runs ({tag}): {len(ok)} ok, {len(failed)} failed" + (f" ({failed['status'].value_counts().to_dict()})" if len(failed) else "")]
    if len(ok):
        lines.append("gate pass (cumulative, in order crash/tilt/outward): " + ", ".join(f"tier>={k}: {(ok['tier'] >= k).sum()}" for k in (1, 2, 3)))
        fmt = lambda d: d[SHORT].round(3).to_string(index=False)
        lines += ["", f"top {args.top} by tier then score:", fmt(ok.head(args.top))]
        lines += ["", "best per scheme:", fmt(ok.groupby("scheme", sort=False).head(1).head(20))]
    print("\n".join(lines))
    if not len(ok):
        return

    table = ok[SHORT + ["batch", "params", "wall_seconds"]].round(3)
    (out / f"results_{tag}.md").write_text(_markdown(table))

    schemes = sorted(ok["scheme"].unique())
    colors = {s: plt.cm.tab10(i % 10) for i, s in enumerate(schemes)}

    def scatter(xcol, fname, xlabel):
        fig, ax = plt.subplots(figsize=(7, 5))
        for s in schemes:
            d = ok[ok["scheme"] == s]
            ax.scatter(d[xcol], d["reaches_per_ep"], color=colors[s], label=s, s=22, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("reaches per episode")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=110)
        plt.close(fig)

    scatter("hover_tilt_frac", f"reaches_vs_tilt_{tag}.png", "hover tilt fraction (slow steps tilted > 15 deg)")
    scatter("outward_frac", f"reaches_vs_outward_{tag}.png", "outward fraction (both nozzles >= 10 deg out)")

    best_score = ok.groupby("scheme")["score"].max().sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(best_score) + 1.5))
    ax.barh(best_score.index, best_score.values, color=[colors[s] for s in best_score.index])
    ax.set_xlabel("best score")
    fig.tight_layout()
    fig.savefig(out / f"best_per_scheme_{tag}.png", dpi=110)
    plt.close(fig)

    order = [b for b in dict.fromkeys(df["batch"]) if b in set(ok["batch"])]
    prog = ok.groupby("batch")["score"].agg(["max", "mean"]).reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prog.index, prog["max"], marker="o", label="best")
    ax.plot(prog.index, prog["mean"], marker="s", label="mean")
    ax.set_ylabel("score")
    ax.set_xlabel("batch")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"score_by_batch_{tag}.png", dpi=110)
    plt.close(fig)


def _markdown(df):
    """Plain markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
