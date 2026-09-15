import numpy as np
import matplotlib.pyplot as plt

# X values: N = 2^i, i = 1,...,8
N = 2 ** np.arange(1, 9)

datasets = [
    {
        "time": [1.65, 1.46, 1.4, 1.44, 1.81, 2.78, 4.54, 8.33],
        "throughput": [0.036, 0.082, 0.17, 0.33, 0.53, 0.69, 0.84, 0.92],
        "title": "recovery of T easures, T=n/2, shard length 64 B"
    }
]

def do_plot(dataset):
    fig, ax_time = plt.subplots(figsize=(8, 5))

    # Left Y axis: time
    line1 = ax_time.plot(
        N, dataset["time"],
        marker="o",
        color="tab:blue",
        label="Time"
    )

    ax_time.set_xlabel("n")
    ax_time.set_ylabel("Time, µs", color="tab:blue")
    ax_time.tick_params(axis="y", labelcolor="tab:blue")

    # Right Y axis: throughput
    ax_throughput = ax_time.twinx()

    line2 = ax_throughput.plot(
        N, dataset["throughput"],
        marker="s",
        color="tab:red",
        label="Throughput"
    )

    ax_throughput.set_ylabel("Throughput, GiB/s", color="tab:red")
    ax_throughput.tick_params(axis="y", labelcolor="tab:red")

    # Since N doubles at every step, use a logarithmic X axis
    ax_time.set_xscale("log", base=2)
    ax_time.set_xticks(N)
    ax_time.set_xticklabels([str(n) for n in N])

    # Combined legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax_time.legend(lines, labels, loc="best")

    ax_time.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_time.set_title(dataset["title"])
    fig.tight_layout()

    plt.show()

for dataset in datasets:
    do_plot(dataset)
