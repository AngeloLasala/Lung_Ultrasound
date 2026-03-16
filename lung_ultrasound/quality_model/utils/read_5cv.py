"""
Read the 5CV file ad plot the results for comparison
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MAIN_PATH = "/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/results/Extrapolates_frames_v2/UNet"          # root folder containing CV result JSONs

TIMESTAMPS = {
    "14-03-2026_11-27": 'only_lung',
    "15-03-2026_08-24": '+_liver',
    "15-03-2026_08-29": '+_plax',
    "15-03-2026_10-32": '+_liver_plax',
}


# ───────────────────────────────────────────────────────────────────────────────


def load_cv_result(main_path: str, timestamp: str) -> dict | None:
    """Search for a file named test_5cv_{timestamp}.json under main_path."""
    target = f"test_5cv_{timestamp}.json"
    for root, _, files in os.walk(main_path):
        if target in files:
            fpath = os.path.join(root, target)
            with open(fpath) as f:
                return json.load(f)
    print(f"[WARNING] File not found: {target}")
    return None


def extract_metrics(data: dict) -> dict:
    """Pull pleura and ribs mean ± std from the summary block."""
    summary = data.get("summary", {})
    return {
        "pleura_mean": summary["pleura_dice"]["mean"],
        "pleura_std":  summary["pleura_dice"]["std"],
        "ribs_mean":   summary["ribs_dice"]["mean"],
        "ribs_std":    summary["ribs_dice"]["std"],
    }


def plot_cv_comparison(main_path: str, timestamps: list[str]) -> None:
    records = {}
    for ts in timestamps.keys():
        data = load_cv_result(main_path, ts)
        if data is not None:
            records[ts] = extract_metrics(data)

    if not records:
        print("No data loaded – check MAIN_PATH and TIMESTAMPS.")
        return

    labels        = list(records.keys())
    labels_name   = [TIMESTAMPS[i] for i in labels]
    dice_mean     = np.array([records[ts]["pleura_mean"] for ts in labels])
    pleura_means  = np.array([records[ts]["pleura_mean"] for ts in labels])
    pleura_stds   = np.array([records[ts]["pleura_std"]  for ts in labels])
    ribs_means    = np.array([records[ts]["ribs_mean"]   for ts in labels])
    ribs_stds     = np.array([records[ts]["ribs_std"]    for ts in labels])

    x      = np.arange(len(labels))
    width  = 0.35
    color_pleura = "tomato"
    color_ribs   = "cornflowerblue"

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2.2), 5))

    bars_p = ax.bar(x - width / 2, pleura_means, width,
                    yerr=pleura_stds, capsize=5,
                    color=color_pleura, alpha=0.85,
                    error_kw=dict(elinewidth=1.2, ecolor="black"))

    bars_r = ax.bar(x + width / 2, ribs_means, width,
                    yerr=ribs_stds, capsize=5,
                    color=color_ribs, alpha=0.85,
                    error_kw=dict(elinewidth=1.2, ecolor="black"))

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Dice score", fontsize=12)
    ax.set_title("Cross-validation results: pleura vs ribs dice", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_name, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color=color_pleura, alpha=0.85, label="Pleura dice"),
        mpatches.Patch(color=color_ribs,   alpha=0.85, label="Ribs dice"),
    ]
    ax.legend(handles=legend_handles, fontsize=11)

    # value labels on top of each bar
    for bar, std in zip(bars_p, pleura_stds):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + std + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, color=color_pleura)

    for bar, std in zip(bars_r, ribs_stds):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + std + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, color=color_ribs)

    plt.tight_layout()
    out_path = os.path.join(main_path, "cv_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_cv_comparison(MAIN_PATH, TIMESTAMPS)