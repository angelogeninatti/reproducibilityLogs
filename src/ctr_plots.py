import pandas as pd
import matplotlib.pyplot as plt
import os

def compute_ctr_at_k(df, max_k=None):
    """Compute CTR@k for a given dataframe."""
    df['clicked_rank'] = df['clicked_document_index']

    if max_k is None:
        max_k = int(df['clicked_rank'].max())

    ctr_values = []
    for k in range(max_k + 1):
        # Only consider impressions that include at least rank k
        impressions_with_k = df[df['query_results'].apply(lambda x: len(str(x).split(',')) > k)]
        # Count clicks that happened at rank k
        clicks_at_k = (impressions_with_k['clicked_rank'] == k).sum()
        ctr = clicks_at_k / len(impressions_with_k) if len(impressions_with_k) > 0 else 0
        ctr_values.append(ctr)
    return ctr_values


def plot_ctr_curves(config, plot_type="line", save_path="ctr_plot.pdf"):
    """
    Plot CTR@k curves (line or bar) for given experiments.

    config = {
        "title": "CTR@k Comparison Across Systems",
        "experiments": {
            "System A": "path/to/system_a.csv",
            "System B": "path/to/system_b.csv",
        }
    }

    plot_type: "line" or "bar"
    save_path: output PDF filename
    """
    experiments = config.get("experiments", {})
    plot_title = config.get("title", "CTR@k Comparison")

    if not experiments:
        print("No experiments provided.")
        return

    plt.figure(figsize=(4, 4))
    max_rank_overall = 0
    ctr_data = {}

    # --- Compute CTR@k for all experiments ---
    for label, file_path in experiments.items():
        df = pd.read_csv(file_path)
        ctr_at_k = compute_ctr_at_k(df)
        ctr_data[label] = ctr_at_k
        max_rank_overall = max(max_rank_overall, len(ctr_at_k))

    ranks = range(1, max_rank_overall + 1)

    # --- Plot CTR@k ---
    if plot_type == "bar":
        bar_width = 0.8 / len(experiments)
        for i, (label, ctr_values) in enumerate(ctr_data.items()):
            plt.bar(
                [r + i * bar_width for r in ranks],
                ctr_values,
                width=bar_width,
                label=label
            )
        plt.xticks([r + bar_width * (len(experiments) / 2) for r in ranks], ranks)
    else:  # default: line plot
        for label, ctr_values in ctr_data.items():
            plt.plot(ranks, ctr_values, marker='o', label=label)

    # --- Finalize plot ---
    plt.title(plot_title)
    plt.xlabel('Rank k')
    plt.ylabel('Click-Through Rate (CTR)')
    plt.legend(title="Click data")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Ensure only integer x-ticks
    plt.xticks(ranks)  # integer-only x-axis ticks

    # --- Save plot as PDF ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf')
    # print(f"Plot saved as {save_path}")
    # plt.show()


if __name__ == "__main__":
    config = {
        "title": "CTR@k comparison of the logged clicks",
        "experiments": {
            "Sample 1": "./data/logs/sampled/confirm_choose_logs_sampled_1.csv",
            "Sample 2": "./data/logs/sampled/confirm_choose_logs_sampled_2.csv",
            "Sample 3": "./data/logs/sampled/confirm_choose_logs_sampled_3.csv",
            "Sample 4": "./data/logs/sampled/confirm_choose_logs_sampled_4.csv",
            "Sample 5": "./data/logs/sampled/confirm_choose_logs_sampled_5.csv",
        }
    }
    plot_ctr_curves(config, plot_type="line", save_path="./results/figures/ctr_logs.pdf")

    config = {
        "title": "CTR@k comparison for sample 1",
        "experiments": {
            "User Logs": "./data/logs/sampled/confirm_choose_logs_sampled_1.csv",
            "DCTR": "./data/simulations/dctr/confirm_choose_logs_sampled_1.csv",
            "DCM": "./data/simulations/dcm/confirm_choose_logs_sampled_1.csv",
            "DBN": "./data/simulations/dbn/confirm_choose_logs_sampled_1.csv",
            "PBM": "./data/simulations/pbm/confirm_choose_logs_sampled_1.csv",
        }
    }
    plot_ctr_curves(config, plot_type="line", save_path="./results/figures/ctr_sample_1.pdf")

    config = {
        "title": "CTR@k comparison for sample 2",
        "experiments": {
            "User Logs": "./data/logs/sampled/confirm_choose_logs_sampled_2.csv",
            "DCTR": "./data/simulations/dctr/confirm_choose_logs_sampled_2.csv",
            "DCM": "./data/simulations/dcm/confirm_choose_logs_sampled_2.csv",
            "DBN": "./data/simulations/dbn/confirm_choose_logs_sampled_2.csv",
            "PBM": "./data/simulations/pbm/confirm_choose_logs_sampled_2.csv",
        }
    }
    plot_ctr_curves(config, plot_type="line", save_path="./results/figures/ctr_sample_2.pdf")

    config = {
        "title": "CTR@k comparison for sample 3",
        "experiments": {
            "User Logs": "./data/logs/sampled/confirm_choose_logs_sampled_3.csv",
            "DCTR": "./data/simulations/dctr/confirm_choose_logs_sampled_3.csv",
            "DCM": "./data/simulations/dcm/confirm_choose_logs_sampled_3.csv",
            "DBN": "./data/simulations/dbn/confirm_choose_logs_sampled_3.csv",
            "PBM": "./data/simulations/pbm/confirm_choose_logs_sampled_3.csv",
        }
    }
    plot_ctr_curves(config, plot_type="line", save_path="./results/figures/ctr_sample_3.pdf")

    config = {
        "title": "CTR@k comparison for sample 4",
        "experiments": {
            "User Logs": "./data/logs/sampled/confirm_choose_logs_sampled_4.csv",
            "DCTR": "./data/simulations/dctr/confirm_choose_logs_sampled_4.csv",
            "DCM": "./data/simulations/dcm/confirm_choose_logs_sampled_4.csv",
            "DBN": "./data/simulations/dbn/confirm_choose_logs_sampled_4.csv",
            "PBM": "./data/simulations/pbm/confirm_choose_logs_sampled_4.csv",
        }
    }
    plot_ctr_curves(config, plot_type="line", save_path="./results/figures/ctr_sample_4.pdf")

    config = {
        "title": "CTR@k comparison for sample 5",
        "experiments": {
            "User Logs": "./data/logs/sampled/confirm_choose_logs_sampled_5.csv",
            "DCTR": "./data/simulations/dctr/confirm_choose_logs_sampled_5.csv",
            "DCM": "./data/simulations/dcm/confirm_choose_logs_sampled_5.csv",
            "DBN": "./data/simulations/dbn/confirm_choose_logs_sampled_5.csv",
            "PBM": "./data/simulations/pbm/confirm_choose_logs_sampled_5.csv",
        }
    }
    plot_ctr_curves(config, plot_type="line", save_path="./results/figures/ctr_sample_5.pdf")
