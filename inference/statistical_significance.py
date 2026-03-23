import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


BASELINE_ROOT = "./training_results/results_multi_run"
DDT_ROOT = "./training_results/results_multi_run_ddt"
OUTPUT_DIR = "./training_results/statistical_tests"
SUPPORTED_DATASETS = ("sepsis", "aki", "ferritin")
KEY_COLUMNS = ["run_id", "ratio"]
METRIC_COLUMNS = ["f1", "auroc", "sensitivity", "cost", "avg_tests", "reward"]
DEFAULT_EQUIVALENCE_MARGINS = {
    "f1": 0.03,
    "auroc": 0.02,
    "sensitivity": 0.03,
}


@dataclass
class TestResult:
    statistic: float
    p_value: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paired significance tests for results_multi_run vs results_multi_run_ddt."
    )
    parser.add_argument(
        "--dataset",
        choices=("all",) + SUPPORTED_DATASETS,
        default="all",
        help="Dataset to evaluate. Defaults to all available datasets.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--f1-margin", type=float, default=DEFAULT_EQUIVALENCE_MARGINS["f1"])
    parser.add_argument("--auroc-margin", type=float, default=DEFAULT_EQUIVALENCE_MARGINS["auroc"])
    parser.add_argument(
        "--sensitivity-margin",
        type=float,
        default=DEFAULT_EQUIVALENCE_MARGINS["sensitivity"],
    )
    return parser.parse_args()


def metric_margins_from_args(args):
    return {
        "f1": args.f1_margin,
        "auroc": args.auroc_margin,
        "sensitivity": args.sensitivity_margin,
    }


def dataset_file(root_dir, dataset):
    return os.path.join(root_dir, dataset, "all_runs_metrics.csv")


def available_datasets(requested_dataset):
    if requested_dataset != "all":
        return [requested_dataset]

    datasets = []
    for dataset in SUPPORTED_DATASETS:
        baseline_path = dataset_file(BASELINE_ROOT, dataset)
        ddt_path = dataset_file(DDT_ROOT, dataset)
        if os.path.exists(baseline_path) and os.path.exists(ddt_path):
            datasets.append(dataset)
    return datasets


def load_paired_dataset(dataset):
    baseline_path = dataset_file(BASELINE_ROOT, dataset)
    ddt_path = dataset_file(DDT_ROOT, dataset)

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Missing baseline metrics file: {baseline_path}")
    if not os.path.exists(ddt_path):
        raise FileNotFoundError(f"Missing DDT metrics file: {ddt_path}")

    baseline_df = pd.read_csv(baseline_path)
    ddt_df = pd.read_csv(ddt_path)

    baseline_df = baseline_df[KEY_COLUMNS + METRIC_COLUMNS].copy()
    ddt_df = ddt_df[KEY_COLUMNS + METRIC_COLUMNS].copy()

    paired_df = baseline_df.merge(
        ddt_df,
        on=KEY_COLUMNS,
        suffixes=("_baseline", "_ddt"),
        how="inner",
        validate="one_to_one",
    )
    paired_df = paired_df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    if paired_df.empty:
        raise ValueError(f"No paired rows found for dataset '{dataset}'.")

    return paired_df


def confidence_interval(diff_values, alpha):
    sample = np.asarray(diff_values, dtype=float)
    n = len(sample)
    mean_diff = float(np.mean(sample))
    if n < 2:
        return mean_diff, mean_diff

    std_diff = float(np.std(sample, ddof=1))
    if std_diff == 0.0:
        return mean_diff, mean_diff

    sem = std_diff / np.sqrt(n)
    t_crit = float(student_t.ppf(1 - alpha / 2, df=n - 1))
    return mean_diff - t_crit * sem, mean_diff + t_crit * sem


def paired_t_test(diff_values, alternative="two-sided", mu0=0.0):
    sample = np.asarray(diff_values, dtype=float) - mu0
    n = len(sample)
    mean_value = float(np.mean(sample))

    if n < 2:
        return TestResult(statistic=np.nan, p_value=np.nan)

    std_value = float(np.std(sample, ddof=1))
    if std_value == 0.0:
        if alternative == "two-sided":
            p_value = 1.0 if np.isclose(mean_value, 0.0) else 0.0
        elif alternative == "less":
            p_value = 1.0 if mean_value >= 0.0 else 0.0
        else:
            p_value = 1.0 if mean_value <= 0.0 else 0.0
        return TestResult(statistic=np.inf if mean_value > 0 else -np.inf, p_value=p_value)

    sem = std_value / np.sqrt(n)
    t_stat = mean_value / sem
    cdf_value = float(student_t.cdf(t_stat, df=n - 1))

    if alternative == "less":
        p_value = cdf_value
    elif alternative == "greater":
        p_value = 1.0 - cdf_value
    else:
        p_value = 2.0 * min(cdf_value, 1.0 - cdf_value)

    return TestResult(statistic=t_stat, p_value=p_value)


def safe_wilcoxon(diff_values, alternative):
    sample = np.asarray(diff_values, dtype=float)
    if len(sample) == 0 or np.allclose(sample, 0.0):
        return TestResult(statistic=np.nan, p_value=np.nan)

    result = wilcoxon(sample, alternative=alternative, zero_method="wilcox", correction=False)
    return TestResult(statistic=float(result.statistic), p_value=float(result.pvalue))


def better_worse_test(diff_values, alpha, metric_type="performance"):
    """
    Determine if DDT is significantly better, worse, or neither compared to baseline.
    
    Args:
        diff_values: ddt_values - baseline_values
        alpha: significance level
        metric_type: 'performance' (higher is better) or 'cost' (lower is better)
    
    Returns:
        Tuple: (test_better, test_worse, direction_label)
        - direction_label: 'better', 'worse', or 'no_significant_difference'
    """
    if metric_type == "performance":
        # For performance metrics: higher DDT is better
        test_better = paired_t_test(diff_values, alternative="greater")  # DDT > baseline
        test_worse = paired_t_test(diff_values, alternative="less")     # DDT < baseline
    else:  # cost
        # For cost: lower DDT is better
        test_better = paired_t_test(diff_values, alternative="less")    # DDT < baseline (lower cost)
        test_worse = paired_t_test(diff_values, alternative="greater")  # DDT > baseline (higher cost)
    
    is_better = not np.isnan(test_better.p_value) and test_better.p_value < alpha
    is_worse = not np.isnan(test_worse.p_value) and test_worse.p_value < alpha
    
    if is_better:
        direction = "better"
    elif is_worse:
        direction = "worse"
    else:
        direction = "no_significant_difference"
    
    return test_better, test_worse, direction


def build_group_rows(dataset, group_label, paired_df, alpha, equivalence_margins):
    rows = []

    # Compare performance metrics (higher is better)
    for metric in ["f1", "auroc", "sensitivity"]:
        baseline_values = paired_df[f"{metric}_baseline"].to_numpy(dtype=float)
        ddt_values = paired_df[f"{metric}_ddt"].to_numpy(dtype=float)
        diff_values = ddt_values - baseline_values

        ci_low, ci_high = confidence_interval(diff_values, alpha)
        test_better, test_worse, direction = better_worse_test(diff_values, alpha, metric_type="performance")
        paired_two_sided = paired_t_test(diff_values, alternative="two-sided")
        wilcoxon_two_sided = safe_wilcoxon(diff_values, alternative="two-sided")

        rows.append(
            {
                "dataset": dataset,
                "group": group_label,
                "metric": metric,
                "n_pairs": len(diff_values),
                "baseline_mean": float(np.mean(baseline_values)),
                "ddt_mean": float(np.mean(ddt_values)),
                "mean_diff_ddt_minus_baseline": float(np.mean(diff_values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ddt_direction": direction,
                "paired_t_pvalue_better": test_better.p_value,
                "paired_t_statistic_better": test_better.statistic,
                "paired_t_pvalue_worse": test_worse.p_value,
                "paired_t_statistic_worse": test_worse.statistic,
                "wilcoxon_pvalue": wilcoxon_two_sided.p_value,
                "wilcoxon_statistic": wilcoxon_two_sided.statistic,
            }
        )

    # Compare cost metric (lower is better)
    cost_values_baseline = paired_df["cost_baseline"].to_numpy(dtype=float)
    cost_values_ddt = paired_df["cost_ddt"].to_numpy(dtype=float)
    cost_diff = cost_values_ddt - cost_values_baseline
    ci_low, ci_high = confidence_interval(cost_diff, alpha)
    test_better, test_worse, direction = better_worse_test(cost_diff, alpha, metric_type="cost")
    paired_two_sided = paired_t_test(cost_diff, alternative="two-sided")
    wilcoxon_two_sided = safe_wilcoxon(cost_diff, alternative="two-sided")

    rows.append(
        {
            "dataset": dataset,
            "group": group_label,
            "metric": "cost",
            "n_pairs": len(cost_diff),
            "baseline_mean": float(np.mean(cost_values_baseline)),
            "ddt_mean": float(np.mean(cost_values_ddt)),
            "mean_diff_ddt_minus_baseline": float(np.mean(cost_diff)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ddt_direction": direction,
            "paired_t_pvalue_better": test_better.p_value,
            "paired_t_statistic_better": test_better.statistic,
            "paired_t_pvalue_worse": test_worse.p_value,
            "paired_t_statistic_worse": test_worse.statistic,
            "wilcoxon_pvalue": wilcoxon_two_sided.p_value,
            "wilcoxon_statistic": wilcoxon_two_sided.statistic,
        }
    )

    return rows


def run_tests(dataset, alpha, equivalence_margins):
    paired_df = load_paired_dataset(dataset)

    rows = []
    rows.extend(build_group_rows(dataset, "overall", paired_df, alpha, equivalence_margins))

    for ratio, ratio_df in paired_df.groupby("ratio", sort=True):
        group_label = f"ratio_{ratio}"
        rows.extend(build_group_rows(dataset, group_label, ratio_df, alpha, equivalence_margins))

    return pd.DataFrame(rows)


def write_outputs(result_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_path = os.path.join(OUTPUT_DIR, "paired_significance_tests.csv")
    result_df.to_csv(overall_path, index=False)

    summary_lines = []
    for dataset in sorted(result_df["dataset"].unique()):
        summary_lines.append(f"Dataset: {dataset}")
        dataset_df = result_df[(result_df["dataset"] == dataset) & (result_df["group"] == "overall")]
        for _, row in dataset_df.iterrows():
            direction = row.get("ddt_direction", "unknown")
            summary_lines.append(
                f"  {row['metric']}: "
                f"baseline={row['baseline_mean']:.4f}, ddt={row['ddt_mean']:.4f}, "
                f"diff={row['mean_diff_ddt_minus_baseline']:.4f}, "
                f"p_better={row['paired_t_pvalue_better']:.4g}, "
                f"p_worse={row['paired_t_pvalue_worse']:.4g}, "
                f"result={direction}"
            )
        summary_lines.append("")

    summary_path = os.path.join(OUTPUT_DIR, "paired_significance_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines).rstrip() + "\n")

    return overall_path, summary_path


def main():
    args = parse_args()
    datasets = available_datasets(args.dataset)
    if not datasets:
        raise FileNotFoundError("No paired datasets found under results_multi_run and results_multi_run_ddt.")

    equivalence_margins = metric_margins_from_args(args)
    result_frames = []
    for dataset in datasets:
        print(f"Running paired tests for {dataset}...")
        result_frames.append(run_tests(dataset, args.alpha, equivalence_margins))

    result_df = pd.concat(result_frames, ignore_index=True)
    csv_path, summary_path = write_outputs(result_df)
    print(f"Saved paired test table to: {os.path.abspath(csv_path)}")
    print(f"Saved paired test summary to: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()