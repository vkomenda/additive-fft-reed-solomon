#!/usr/bin/env python3
"""Plot Criterion results as static SVG charts.

Usage (after `cargo bench`):  python scripts/plot.py
Writes docs/charts/*.svg and prints Markdown for the README.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CRITERION = ROOT / "target" / "criterion"
OUT = ROOT / "docs" / "charts"

FEATURES = ["gfni", "avx2", "neon"]    # plotted features, in legend and colour order
SPECIES = ["cantor", "standard"]       # one chart per species
ALIGNMENTS = ["aligned", "unaligned"]  # one chart per alignment

_F, _S, _A = (f"({'|'.join(x)})" for x in (FEATURES, SPECIES, ALIGNMENTS))
# {feature}_{species}_N{n}_T{t}_{alignment}         x is n
SHARDED_RE = re.compile(rf"{_F}_{_S}_N(\d+)_T(\d+)_{_A}")
# {feature}_{species}_{method}_k{k}_{alignment}     x is k, one chart per method
TRANSFORM_RE = re.compile(rf"{_F}_{_S}_(.+)_k(\d+)_{_A}")

GROUP_TITLES = {
    "encode_systematic_sharded": "systematic encoding of T parity shards from T data shards",
    "recover_erasures_sharded_clobber": "recovery of random T erased shards (Leopard-style)",
    "recover_erasures_sharded": "recovery of random T erased shards (LNH original)",
    "transform": "transform",
}

plt.rcParams["svg.hashsalt"] = "criterion"  # stable SVG ids -> small git diffs


def load(path):
    return json.loads(path.read_text())


def parse(group, function_id):
    """-> (chart_suffix, x, x_label, feature, note) or None to skip."""
    if group == "transform":
        m = TRANSFORM_RE.fullmatch(function_id)
        if not m:
            return None
        feature, species, method, k, alignment = m[1], m[2], m[3], int(m[4]), m[5]
        return (method, species, alignment), k, "k", feature, ""
    m = SHARDED_RE.fullmatch(function_id)
    if not m:
        return None
    feature, species, n, t, alignment = m[1], m[2], int(m[3]), int(m[4]), m[5]
    if n != 2 * t:
        return None
    return (species, alignment), n, "n", feature, "T=n/2"


def collect():
    """{(group, chart_suffix, shard, x_label, note): {feature: [(x, µs, GiB/s)]}}"""
    charts = defaultdict(lambda: defaultdict(list))
    for meta_path in CRITERION.glob("**/new/benchmark.json"):
        meta = load(meta_path)
        tp = meta.get("throughput") or {}
        nbytes = tp.get("Bytes", tp.get("BytesDecimal"))
        parsed = parse(meta["group_id"], meta["function_id"])
        if nbytes is None or parsed is None:
            continue
        suffix, x, x_label, feature, note = parsed
        if feature not in FEATURES:
            continue
        est = load(meta_path.with_name("estimates.json"))
        ns = (est.get("slope") or est["mean"])["point_estimate"]  # as Criterion reports
        shard = int(meta["value_str"])
        charts[meta["group_id"], suffix, shard, x_label, note][feature].append(
            (x, ns / 1e3, nbytes / (ns * 1e-9) / 2**30)
        )
    return charts


def plot(title, subtitle, series, x_label, path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax2 = ax.twinx()
    lines = []
    for feature, color in zip(FEATURES, plt.cm.tab10.colors):
        if feature not in series:
            continue
        x, time, thr = zip(*sorted(series[feature]))
        lines += ax.plot(x, time, "o-", color=color, label=f"{feature}, time")
        lines += ax2.plot(x, thr, "s--", color=color, label=f"{feature}, throughput")

    xs = sorted({p[0] for pts in series.values() for p in pts})
    ax.set_xlabel(x_label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(v) for v in xs])
    ax.minorticks_off()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylabel("Time, µs")
    ax2.set_ylabel("Throughput, GiB/s")

    ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize="small")
    ax.set_title(subtitle, fontsize=9.5, color="0.35")  # second line, below the title
    fig.suptitle(title, fontsize=13, y=0.975)
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def main():
    charts = collect()
    if not charts:
        raise SystemExit(f"No results under {CRITERION}; run `cargo bench` first.")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.svg"):
        old.unlink()
    for (group, suffix, shard, x_label, note), series in sorted(charts.items()):
        size = f"{shard >> 10} KiB" if shard >= 1024 else f"{shard} B"
        title = GROUP_TITLES.get(group, group)
        subtitle = ", ".join(filter(None, (*suffix, note, f"shard length {size}")))
        path = OUT / f"{'_'.join((group, *suffix, str(shard)))}.svg"
        plot(title, subtitle, series, x_label, path)
        print(f"![{title}]({path.relative_to(ROOT).as_posix()})")

if __name__ == "__main__":
    main()
