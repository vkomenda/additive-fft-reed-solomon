#!/usr/bin/env python3
"""Plot Criterion results as static SVG charts.

Usage (after `cargo bench`):  python scripts/plot.py
Writes docs/charts/<group>_<feature>_<shard_len>.svg and prints Markdown for the README.
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
FUNC_RE = re.compile(r"N(\d+)_T(\d+)_(\w+)_aligned")
FEAT = ["gfni", "avx2"]
GROUP_TITLES = {
    "encode_systematic_sharded": "systematic encoding of T parity shards from T data shards",
    "recover_erasures_sharded": "recovery of random T erased shards (LNH original)",
    "recover_erasures_sharded_clobber": "recovery of random T erased shards (faster algo)"
}

plt.rcParams["svg.hashsalt"] = "criterion"

def load(path):
    return json.loads(path.read_text())

def collect():
    groups = defaultdict(list)
    for meta_path in CRITERION.glob("**/new/benchmark.json"):
        meta = load(meta_path)
        m = FUNC_RE.fullmatch(meta["function_id"])
        tp = meta.get("throughput") or {}
        nbytes = tp.get("Bytes", tp.get("BytesDecimal"))
        if not m or nbytes is None:
            continue
        est = load(meta_path.with_name("estimates.json"))
        ns = (est.get("slope") or est["mean"])["point_estimate"]
        n, t, feat = int(m[1]), int(m[2]), m[3]
        if n != 2 * t or feat not in FEAT:
            continue
        groups[meta["group_id"], feat, int(meta["value_str"])].append(
            (n, t, ns / 1e3, nbytes / (ns * 1e-9) / 2**30)
        )
    return groups

def plot(title, pts, path):
    n, _, time, thr = zip(*sorted(pts))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")

    lines = ax.plot(n, time, "o-", color="tab:blue", label="Time")
    ax.set_xlabel("n")
    ax.set_ylabel("Time, µs", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [str(v) for v in n])
    ax.minorticks_off()
    ax.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax.twinx()
    lines += ax2.plot(n, thr, "s--", color="tab:red", label="Throughput")
    ax2.set_ylabel("Throughput, GiB/s", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax.legend(lines, [l.get_label() for l in lines], loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)

def main():
    groups = collect()
    if not groups:
        raise SystemExit(f"No results under {CRITERION}; run `cargo bench` first.")
    OUT.mkdir(parents=True, exist_ok=True)
    for (group, feat, shard), pts in sorted(groups.items()):
        size = f"{shard >> 10} KiB" if shard >= 1024 else f"{shard} B"
        title = f"{GROUP_TITLES.get(group, group)} ({feat}), T=n/2, shard length {size}"
        path = OUT / f"{group}_{feat}_{shard}.svg"
        plot(title, pts, path)
        print(f"![{title}]({path.relative_to(ROOT).as_posix()})")


if __name__ == "__main__":
    main()
