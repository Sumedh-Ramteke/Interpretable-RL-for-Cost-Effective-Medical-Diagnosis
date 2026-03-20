import itertools
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from data_preprocessing.blood_panel_data_preprocessing import aki_data, ferritin_data, sepsis_data
from data_preprocessing.data_loader import Data_Loader

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


NUM_RUNS = 10
SUPPORTED_MODES = ["sepsis", "aki", "ferritin", "all"]
MAX_TESTS_BY_MODE = {
    "sepsis": 5,
    "aki": 4,
    "ferritin": 3,
}


def ask_mode() -> str:
    print("Select dataset mode:")
    print("1. sepsis")
    print("2. aki")
    print("3. ferritin")
    print("4. all")

    mapping = {"1": "sepsis", "2": "aki", "3": "ferritin", "4": "all"}
    while True:
        choice = input("Enter choice [1/2/3/4]: ").strip().lower()
        if choice in mapping:
            return mapping[choice]
        if choice in SUPPORTED_MODES:
            return choice
        print("Invalid choice. Please enter one of: 1, 2, 3, 4, sepsis, aki, ferritin, all")


def ask_num_runs(default_runs: int = NUM_RUNS) -> int:
    while True:
        raw = input(f"Number of runs [default {default_runs}]: ").strip()
        if raw == "":
            return default_runs
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Invalid number. Enter a positive integer.")


def print_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    if df.empty:
        print("No rows to display.")
        return
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def load_dataset(mode: str) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    if mode == "sepsis":
        return sepsis_data()
    if mode == "aki":
        return aki_data()
    if mode == "ferritin":
        return ferritin_data()
    raise ValueError(f"Unsupported mode: {mode}")


def safe_predict_proba(clf: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1]

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))

    preds = clf.predict(x)
    return preds.astype(float)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob > threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "f1": f1,
        "auroc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "fnr": fnr,
    }


def create_model(model_class: Any, params: dict[str, Any], extra_params: dict[str, Any], random_state: int):
    init_kwargs = dict(params)
    init_kwargs.update(extra_params)

    try:
        return model_class(**init_kwargs, random_state=random_state)
    except TypeError:
        return model_class(**init_kwargs)


def run_multi_baseline(
    model_class: Any,
    param_grid: dict[str, list[Any]],
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_runs: int,
    extra_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run grid search across multiple independent runs and aggregate mean/std metrics.
    """
    if extra_params is None:
        extra_params = {}

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = [dict(zip(param_names, vals)) for vals in itertools.product(*param_values)]

    all_rows = []
    best_rows = []

    for run_id in range(1, num_runs + 1):
        current_seed = 42
        print(f"--- {model_name}: Run {run_id}/{num_runs} ---")
        run_rows = []

        for params in tqdm(combinations, desc=f"{model_name} run {run_id}", leave=False):
            clf = create_model(model_class, params, extra_params, current_seed)
            clf.fit(x_train, y_train)

            y_val_prob = safe_predict_proba(clf, x_val)
            y_test_prob = safe_predict_proba(clf, x_test)

            val_metrics = binary_metrics(y_val, y_val_prob)
            test_metrics = binary_metrics(y_test, y_test_prob)

            row = {
                "run_id": run_id,
                "hp1": params[param_names[0]] if len(param_names) > 0 else None,
                "hp2": params[param_names[1]] if len(param_names) > 1 else None,
                "param1_name": param_names[0] if len(param_names) > 0 else "",
                "param2_name": param_names[1] if len(param_names) > 1 else "",
                "val_f1": val_metrics["f1"],
                "val_auroc": val_metrics["auroc"],
                "test_f1": test_metrics["f1"],
                "test_auroc": test_metrics["auroc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_specificity": test_metrics["specificity"],
                "test_fnr": test_metrics["fnr"],
            }
            all_rows.append(row)
            run_rows.append(row)

        # Select best hyperparameter configuration within each run using validation F1.
        run_df = pd.DataFrame(run_rows)
        best_idx = run_df["val_f1"].idxmax()
        best_run = run_df.loc[best_idx].to_dict()
        best_run["model"] = model_name
        best_rows.append(best_run)

    full_df = pd.DataFrame(all_rows)

    summary = full_df.groupby(["hp1", "hp2", "param1_name", "param2_name"]).agg(
        {
            "val_f1": ["mean", "std"],
            "val_auroc": ["mean", "std"],
            "test_f1": ["mean", "std"],
            "test_auroc": ["mean", "std"],
            "test_accuracy": ["mean", "std"],
            "test_precision": ["mean", "std"],
            "test_recall": ["mean", "std"],
            "test_specificity": ["mean", "std"],
            "test_fnr": ["mean", "std"],
        }
    )

    summary = summary.reset_index()
    summary.columns = [f"{a}_{b}" if b else a for a, b in summary.columns]
    best_runs_df = pd.DataFrame(best_rows)
    return summary, best_runs_df


def summarize_from_all_runs(all_runs_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "f1",
        "auroc",
        "acc",
        "precision",
        "sensitivity",
        "specificity",
        "fnr",
        "cost",
        "tests",
    ]
    available_metric_cols = [c for c in metric_cols if c in all_runs_df.columns]
    summary = all_runs_df.groupby("model")[available_metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary.reset_index()


def run_for_dataset(mode: str, num_runs: int) -> pd.DataFrame:
    print(f"\n{'=' * 60}\nRunning baselines for: {mode.upper()}\n{'=' * 60}")

    save_dir = f"training_results/baseline/{mode}"
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = f"{save_dir}/all_runs_metrics.csv"
    summary_path = f"{save_dir}/summary_results.csv"

    # If summary already exists, skip training and print saved results.
    if os.path.exists(summary_path):
        cached = pd.read_csv(summary_path)
        print(f"Found existing results for {mode}. Skipping training.")
        print_summary(cached, f"Saved Summary ({mode.upper()})")
        return cached

    data, block, _ = load_dataset(mode)
    max_tests = MAX_TESTS_BY_MODE.get(mode, len(block) - 1)

    data_loader = Data_Loader(data, block, test_ratio=0.2, val_ratio=0.2)

    x_train = data_loader.train[:, :-1]
    x_val = data_loader.val[:, :-1]
    x_test = data_loader.test[:, :-1]

    y_train = data_loader.train[:, -1]
    y_val = data_loader.val[:, -1]
    y_test = data_loader.test[:, -1]

    # Model grids
    rf_grid = {
        "n_estimators": np.linspace(20, 620, 11, dtype=int).tolist(),
        "max_depth": [3, 7, 11, 21, 31, 61],
    }

    lr_grid = {
        "C": [0.0001, 0.001, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
    }

    xgb_grid = {
        "n_estimators": np.linspace(20, 620, 11, dtype=int).tolist(),
        "max_depth": [3, 7, 11, 21, 31, 61],
    }

    lgbm_grid = {
        "n_estimators": np.linspace(20, 620, 11, dtype=int).tolist(),
        "max_depth": [3, 7, 11, 21, 31, 61],
    }

    per_model_best_runs = []

    # Random Forest
    rf_grid_summary, rf_best_runs = run_multi_baseline(
        RandomForestClassifier,
        rf_grid,
        "RandomForest",
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        num_runs,
        extra_params={"class_weight": "balanced"},
    )
    rf_path = f"{save_dir}/RF_{mode}_grid_summary.csv"
    rf_grid_summary.to_csv(rf_path, index=False)
    per_model_best_runs.append(rf_best_runs)

    # Logistic Regression
    lr_grid_summary, lr_best_runs = run_multi_baseline(
        LogisticRegression,
        lr_grid,
        "LogisticRegression",
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        num_runs,
        extra_params={"class_weight": "balanced", "solver": "saga", "max_iter": 1000},
    )
    lr_path = f"{save_dir}/LR_{mode}_grid_summary.csv"
    lr_grid_summary.to_csv(lr_path, index=False)
    per_model_best_runs.append(lr_best_runs)

    # XGBoost (optional)
    if XGBClassifier is not None:
        xgb_grid_summary, xgb_best_runs = run_multi_baseline(
            XGBClassifier,
            xgb_grid,
            "XGBoost",
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            num_runs,
            extra_params={"eval_metric": "logloss"},
        )
        xgb_path = f"{save_dir}/XGB_{mode}_grid_summary.csv"
        xgb_grid_summary.to_csv(xgb_path, index=False)
        per_model_best_runs.append(xgb_best_runs)
    else:
        print("Skipping XGBoost: package not available in environment.")

    # LightGBM (optional)
    if LGBMClassifier is not None:
        lgbm_grid_summary, lgbm_best_runs = run_multi_baseline(
            LGBMClassifier,
            lgbm_grid,
            "LightGBM",
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            num_runs,
            extra_params={"class_weight": "balanced", "verbose": -1},
        )
        lgbm_path = f"{save_dir}/LGBM_{mode}_grid_summary.csv"
        lgbm_grid_summary.to_csv(lgbm_path, index=False)
        per_model_best_runs.append(lgbm_best_runs)
    else:
        print("Skipping LightGBM: package not available in environment.")

    # Multi-train-like outputs: all_runs_metrics.csv + summary_results.csv
    all_runs_df = pd.concat(per_model_best_runs, ignore_index=True)
    all_runs_df["dataset"] = mode
    all_runs_df["cost"] = 5.0
    all_runs_df["tests"] = float(max_tests)
    all_runs_df = all_runs_df.rename(
        columns={
            "test_f1": "f1",
            "test_auroc": "auroc",
            "test_accuracy": "acc",
            "test_precision": "precision",
            "test_recall": "sensitivity",
            "test_specificity": "specificity",
            "test_fnr": "fnr",
        }
    )
    all_runs_df = all_runs_df[
        [
            "run_id",
            "model",
            "f1",
            "auroc",
            "acc",
            "precision",
            "sensitivity",
            "specificity",
            "fnr",
            "cost",
            "tests",
            "hp1",
            "hp2",
            "param1_name",
            "param2_name",
            "dataset",
        ]
    ]
    all_runs_df.to_csv(metrics_path, index=False)

    summary_df = summarize_from_all_runs(all_runs_df)

    summary_df.to_csv(summary_path, index=False)
    print(f"Saved per-run metrics: {metrics_path}")
    print(f"Saved summarized comparison: {summary_path}")
    print_summary(summary_df, f"Summary ({mode.upper()})")

    return summary_df


def main() -> None:
    mode = ask_mode()
    num_runs = ask_num_runs(default_runs=NUM_RUNS)

    if mode == "all":
        overall_path = "training_results/baseline/models_comparison_multi_all_datasets.csv"
        if os.path.exists(overall_path):
            overall_cached = pd.read_csv(overall_path)
            print("Found existing all-dataset summary. Skipping training.")
            print_summary(overall_cached, "Saved Summary (ALL DATASETS)")
            return

        all_summaries = []
        for ds in ["sepsis", "aki", "ferritin"]:
            all_summaries.append(run_for_dataset(ds, num_runs=num_runs))

        overall = pd.concat(all_summaries, ignore_index=True)
        os.makedirs("training_results/baseline", exist_ok=True)
        overall.to_csv(overall_path, index=False)
        print(f"Saved overall all-dataset summary: {overall_path}")
        print_summary(overall, "Summary (ALL DATASETS)")
    else:
        run_for_dataset(mode, num_runs=num_runs)


if __name__ == "__main__":
    main()
