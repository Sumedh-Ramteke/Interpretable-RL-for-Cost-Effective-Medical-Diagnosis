import os

import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURATION ---
DATASETS = ["sepsis", "aki", "ferritin"]
BASE_RESULTS_PATH = "./training_results"

OUTPUT_DIR = "./visualizations/paper_plots_simple"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Color scheme for methods
COLORS = {
    "Existing SM-DDPO Approach": "#1f77b4",  # blue
    "DDT Approach (Ours)": "#d62728",  # red
}


def load_dataset_data(dataset_name):
    """
    Load and combine summary data for both methods for a given dataset.
    Returns a DataFrame with columns: ratio, Method, f1, cost, auroc, sensitivity
    """
    og_path = f"{BASE_RESULTS_PATH}/results_multi_run/{dataset_name}/summary_results.csv"
    ddt_path = f"{BASE_RESULTS_PATH}/results_multi_run_ddt/{dataset_name}/summary_results.csv"

    dfs = []

    # Load SM-DDPO
    if os.path.exists(og_path):
        df_og = pd.read_csv(og_path)
        df_og["Method"] = "Existing SM-DDPO Approach"
        df_og = df_og.rename(
            columns={
                "f1_mean": "f1",
                "cost_mean": "cost",
                "auroc_mean": "auroc",
                "sensitivity_mean": "sensitivity",
            }
        )
        dfs.append(df_og[["ratio", "Method", "f1", "cost", "auroc", "sensitivity"]])
    else:
        print(f"Warning: SM-DDPO file not found at {og_path}")

    # Load DDT
    if os.path.exists(ddt_path):
        df_ddt = pd.read_csv(ddt_path)
        df_ddt["Method"] = "DDT Approach (Ours)"
        df_ddt = df_ddt.rename(
            columns={
                "f1_mean": "f1",
                "cost_mean": "cost",
                "auroc_mean": "auroc",
                "sensitivity_mean": "sensitivity",
            }
        )
        dfs.append(df_ddt[["ratio", "Method", "f1", "cost", "auroc", "sensitivity"]])
    else:
        print(f"Warning: DDT file not found at {ddt_path}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    raise FileNotFoundError(f"No summary files found for {dataset_name}")


def apply_dynamic_legend_above_data(fig, y_values):
    """
    Keep legend inside plot but above the highest y point by adding dynamic y-axis headroom.
    """
    y_min = float(pd.Series(y_values).min())
    y_max = float(pd.Series(y_values).max())

    span = y_max - y_min
    if span <= 0:
        span = max(abs(y_max) * 0.1, 1e-3)

    lower_pad = 0.05 * span
    upper_pad = 0.30 * span

    fig.update_yaxes(range=[y_min - lower_pad, y_max + upper_pad])
    fig.update_layout(
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        )
    )


def plot_cost_vs_ratio(df, dataset_name):
    """
    PLOT 1: Average Costs vs Penalty Ratio
    """
    print(f"  Generating: Avg. Costs vs Penalty Ratio ({dataset_name})...")

    df_sorted = df.sort_values("ratio")

    fig = go.Figure()

    for method in df["Method"].unique():
        df_m = df_sorted[df_sorted["Method"] == method]
        fig.add_trace(
            go.Scatter(
                x=df_m["ratio"],
                y=df_m["cost"],
                mode="lines+markers",
                name=method,
                line=dict(color=COLORS.get(method, "gray"), width=2),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title=f"<b>Average Cost vs Penalty Ratio - {dataset_name.upper()}</b>",
        xaxis_title="Penalty Ratio (λ)",
        yaxis_title="Average Cost ($)",
        height=500,
        width=800,
        hovermode="x unified",
        template="plotly_white",
    )
    apply_dynamic_legend_above_data(fig, df_sorted["cost"])

    output_path = os.path.join(OUTPUT_DIR, dataset_name, "1_cost_vs_ratio.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)


def plot_f1_vs_ratio(df, dataset_name):
    """
    PLOT 2: F1 Score vs Penalty Ratio
    """
    print(f"  Generating: F1 Score vs Penalty Ratio ({dataset_name})...")

    df_sorted = df.sort_values("ratio")

    fig = go.Figure()

    for method in df["Method"].unique():
        df_m = df_sorted[df_sorted["Method"] == method]
        fig.add_trace(
            go.Scatter(
                x=df_m["ratio"],
                y=df_m["f1"],
                mode="lines+markers",
                name=method,
                line=dict(color=COLORS.get(method, "gray"), width=2),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title=f"<b>F1 Score vs Penalty Ratio - {dataset_name.upper()}</b>",
        xaxis_title="Penalty Ratio (λ)",
        yaxis_title="F1 Score",
        height=500,
        width=800,
        hovermode="x unified",
        template="plotly_white",
    )
    apply_dynamic_legend_above_data(fig, df_sorted["f1"])

    output_path = os.path.join(OUTPUT_DIR, dataset_name, "2_f1_vs_ratio.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)


def plot_cost_vs_recall_tradeoff(df, dataset_name):
    """
    PLOT 3: Cost vs Recall (Sensitivity) Tradeoff Curve
    """
    print(f"  Generating: Cost vs Recall Tradeoff ({dataset_name})...")

    fig = go.Figure()

    for method in df["Method"].unique():
        df_m = df[df["Method"] == method].sort_values("cost")
        fig.add_trace(
            go.Scatter(
                x=df_m["cost"],
                y=df_m["sensitivity"],
                mode="lines+markers",
                name=method,
                line=dict(color=COLORS.get(method, "gray"), width=2),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title=f"<b>Cost vs Recall (Sensitivity) Tradeoff - {dataset_name.upper()}</b>",
        xaxis_title="Average Cost ($)",
        yaxis_title="Recall (Sensitivity)",
        height=500,
        width=800,
        hovermode="closest",
        template="plotly_white",
    )
    apply_dynamic_legend_above_data(fig, df["sensitivity"])

    output_path = os.path.join(OUTPUT_DIR, dataset_name, "3_cost_vs_recall.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)


def plot_auroc_vs_ratio(df, dataset_name):
    """
    PLOT 4: AUROC vs Penalty Ratio
    """
    print(f"  Generating: AUROC vs Penalty Ratio ({dataset_name})...")

    df_sorted = df.sort_values("ratio")

    fig = go.Figure()

    for method in df["Method"].unique():
        df_m = df_sorted[df_sorted["Method"] == method]
        fig.add_trace(
            go.Scatter(
                x=df_m["ratio"],
                y=df_m["auroc"],
                mode="lines+markers",
                name=method,
                line=dict(color=COLORS.get(method, "gray"), width=2),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title=f"<b>AUROC vs Penalty Ratio - {dataset_name.upper()}</b>",
        xaxis_title="Penalty Ratio (λ)",
        yaxis_title="AUROC",
        height=500,
        width=800,
        hovermode="x unified",
        template="plotly_white",
    )
    apply_dynamic_legend_above_data(fig, df_sorted["auroc"])

    output_path = os.path.join(OUTPUT_DIR, dataset_name, "4_auroc_vs_ratio.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)


def main():
    """
    Generate all 4 plots for each dataset.
    """
    try:
        for dataset in DATASETS:
            print(f"\nProcessing {dataset.upper()} dataset...")
            df = load_dataset_data(dataset)

            plot_cost_vs_ratio(df, dataset)
            plot_f1_vs_ratio(df, dataset)
            plot_cost_vs_recall_tradeoff(df, dataset)
            plot_auroc_vs_ratio(df, dataset)

        print(f"\n✓ All plots saved to: {os.path.abspath(OUTPUT_DIR)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
