import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from graphviz import Digraph

# Ensure repo root is importable regardless of invocation location.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocessing.blood_panel_data_preprocessing import sepsis_data, aki_data, ferritin_data
from data_preprocessing.data_loader import Data_Loader
from imputer.imputation import Imputer
from classifiers.classifier_ddt import Classifier
from rl_agent.rl_ddt import RL, DDTMaskableActorCriticPolicy


SUPPORTED_DATASETS = ["sepsis", "aki", "ferritin"]


def ask_dataset() -> str:
    """Prompt the user to select a disease dataset."""
    print("Select disease dataset:")
    print("1. sepsis")
    print("2. aki")
    print("3. ferritin")
    raw = input("Enter choice (1/2/3 or name): ").strip().lower()

    by_index = {"1": "sepsis", "2": "aki", "3": "ferritin"}
    if raw in by_index:
        return by_index[raw]
    if raw in SUPPORTED_DATASETS:
        return raw

    raise ValueError(f"Unsupported choice: {raw}. Use one of {SUPPORTED_DATASETS}.")


def load_summaries(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load summary CSVs for both methods for a chosen dataset."""
    sm_path = REPO_ROOT / "training_results" / "results_multi_run" / dataset / "summary_results.csv"
    ddt_path = REPO_ROOT / "training_results" / "results_multi_run_ddt" / dataset / "summary_results.csv"

    if not sm_path.exists():
        raise FileNotFoundError(f"Missing summary file: {sm_path}")
    if not ddt_path.exists():
        raise FileNotFoundError(f"Missing summary file: {ddt_path}")

    df_sm = pd.read_csv(sm_path)
    df_ddt = pd.read_csv(ddt_path)

    if "ratio" not in df_sm.columns or "f1_mean" not in df_sm.columns:
        raise ValueError(f"Expected columns ratio and f1_mean in {sm_path}")
    if "ratio" not in df_ddt.columns or "f1_mean" not in df_ddt.columns:
        raise ValueError(f"Expected columns ratio and f1_mean in {ddt_path}")

    return df_sm, df_ddt


def select_best_common_ratio(dataset: str) -> float:
    """
    Compare summary files on common ratios and pick the ratio where DDT has highest F1.

    We only consider ratios present in both summary files to keep comparisons fair.
    """
    df_sm, df_ddt = load_summaries(dataset)

    merged = pd.merge(
        df_sm[["ratio", "f1_mean"]].rename(columns={"f1_mean": "f1_sm"}),
        df_ddt[["ratio", "f1_mean"]].rename(columns={"f1_mean": "f1_ddt"}),
        on="ratio",
        how="inner",
    )

    if merged.empty:
        raise ValueError(f"No common ratios found between results_multi_run and results_multi_run_ddt for {dataset}")

    merged["f1_pair_avg"] = (merged["f1_sm"] + merged["f1_ddt"]) / 2.0

    # Primary criterion: best DDT F1 on common ratios.
    # Tie-breakers: higher baseline F1, then higher pair average.
    best = merged.sort_values(["f1_ddt", "f1_sm", "f1_pair_avg"], ascending=False).iloc[0]

    print("\nCommon-ratio comparison (summary_results.csv):")
    print(merged.sort_values("ratio").to_string(index=False))
    print(
        f"\nSelected ratio = {best['ratio']} "
        f"(DDT f1={best['f1_ddt']:.4f}, SM-DDPO f1={best['f1_sm']:.4f}, pair_avg={best['f1_pair_avg']:.4f})"
    )

    return float(best["ratio"])


def _ratio_folder_name(ratio: float) -> str:
    """Normalize ratio folder name to existing format (e.g., 3 instead of 3.0 when integral)."""
    if float(ratio).is_integer():
        return str(int(ratio))
    return str(ratio)


def select_best_ddt_run_and_policy(dataset: str, ratio: float) -> tuple[int, str]:
    """Pick the highest-F1 run at the selected ratio and return policy zip path."""
    all_runs_path = REPO_ROOT / "training_results" / "results_multi_run_ddt" / dataset / "all_runs_metrics.csv"
    if not all_runs_path.exists():
        raise FileNotFoundError(f"Missing all-runs file: {all_runs_path}")

    df = pd.read_csv(all_runs_path)
    if not {"run_id", "ratio", "f1"}.issubset(df.columns):
        raise ValueError(f"Expected columns run_id, ratio, f1 in {all_runs_path}")

    ratio_rows = df[np.isclose(df["ratio"].astype(float), float(ratio), rtol=0.0, atol=1e-9)]
    if ratio_rows.empty:
        raise ValueError(f"No DDT runs found for ratio {ratio} in {all_runs_path}")

    best_row = ratio_rows.sort_values("f1", ascending=False).iloc[0]
    run_id = int(best_row["run_id"])

    ratio_name = _ratio_folder_name(ratio)
    run_dir = REPO_ROOT / "training_results" / "results_multi_run_ddt" / dataset / f"run_{run_id}" / f"ratio_{ratio_name}"

    expected = run_dir / f"policy_ratio_{ratio_name}.zip"
    if expected.exists():
        return run_id, str(expected)

    alternatives = sorted(run_dir.glob("policy_ratio*.zip"))
    if alternatives:
        return run_id, str(alternatives[0])

    raise FileNotFoundError(f"No policy zip found in {run_dir}")


def build_feature_names(dim: int) -> list[str]:
    """Return generic feature labels aligned with RL state layout."""
    base = [f"Var_{i}" for i in range(dim)]
    return ["CLF_Prob_Healthy", "CLF_Prob_Sick"] + base + [f"Mask_{x}" for x in base]


def get_actual_feature_names(dataset: str) -> list[str]:
    """
    Extract actual feature names from blood panel dataset by replicating 
    the preprocessing logic in blood_panel_data_preprocessing.py.
    Returns ordered list: [block0_features, block1_features, ..., block_n_features].
    """
    # Map dataset to CSV filename and preprocessing logic
    dataset_map = {
        'sepsis': ('datasets/sm_ddpo_sepsis_dataset.csv', 'sepsis'),
        'aki': ('datasets/aki_dataset.csv', 'aki'),
        'ferritin': ('datasets/ferritin_dataset.csv', 'ferritin'),
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset}")

    filename, mode = dataset_map[dataset]

    # Try loading from current dir or data subdir
    try:
        df = pd.read_csv(filename, na_values=["NULL"])
    except FileNotFoundError:
        try:
            df = pd.read_csv(f"data/{filename}", na_values=["NULL"])
        except FileNotFoundError:
            print(f"Warning: Could not load {filename}. Using generic feature names.")
            return []

    # Remove target and ID columns
    if mode == 'sepsis':
        if 'hospital_expire_flag' in df.columns:
            del df['hospital_expire_flag']
        if 'icustay_id' in df.columns:
            del df['icustay_id']
    elif mode == 'aki':
        if 'label' in df.columns:
            del df['label']
        if 'icustay_id' in df.columns:
            del df['icustay_id']
    elif mode == 'ferritin':
        if 'label' in df.columns:
            del df['label']
        if 'subject_id' in df.columns:
            del df['subject_id']
        if 'hadm_id' in df.columns:
            del df['hadm_id']

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    # Fill NaNs with mean
    df = df.fillna(df.mean())

    # Define Block 0 (Demographics)
    if mode == 'sepsis':
        block0_candidates = [
            'age', 'sofa_score', 'bmi', 'metastatic_cancer', 'diabetes', 'mechanical_ventilation'
        ]
        block0_prefixes = ['gender', 'race', 'marital', 'insurance', 'admission']
    elif mode == 'aki':
        block0_candidates = ['age', 'gender_num', 'race_black']
        block0_prefixes = []
    elif mode == 'ferritin':
        block0_candidates = ['age', 'gender_num']
        block0_prefixes = []

    # Identify block 0 columns
    block0_cols = []
    for c in df.columns:
        if c in block0_candidates:
            block0_cols.append(c)
        elif any(c.startswith(p) for p in block0_prefixes):
            block0_cols.append(c)

    ordered_names = list(block0_cols)  # Add block 0 features

    # Process test blocks
    if mode == 'sepsis':
        panels = [
            ['heart_rate', 'resp_rate', 'sbp', 'dbp', 'map', 'spo2', 'temperature', 'gcs', 'urine_output', 'tidal_volume'],  # Vitals
            ['hemoglobin', 'hematocrit', 'platelet', 'rbc', 'wbc'],  # CBC
            ['glucose', 'bicarbonate', 'creatinine', 'chloride', 'co2', 'sodium', 'potassium', 'bun', 'calcium'],  # CMP
            ['ast', 'bilirubin', 'albumin', 'magnesium'],  # Liver
            ['lactate', 'base_excess', 'ph', 'fio2', 'ptt', 'inr'],  # Gas/Coag
        ]
    elif mode == 'aki':
        panels = [
            ['hemoglobin_min', 'wbc_max', 'platelet_min'],  # CBC
            ['glucose_max', 'bicarbonate_min', 'creatinine_min', 'creatinine_max', 'bun_max', 'calcium_min', 'potassium_max', 'egfr'],  # CMP
            ['ptt_min', 'ptt_max', 'inr_min', 'inr_max', 'pt_min', 'pt_max'],  # APTT
            ['spo2_min', 'spo2_max', 'ph_min', 'ph_max', 'pco2_min', 'pco2_max', 'po2_min', 'po2_max'],  # ABG
        ]
    elif mode == 'ferritin':
        panels = [
            ['hemoglobin', 'wbc', 'platelets'],  # CBC
            ['creatinine', 'bun', 'glucose', 'sodium', 'potassium', 'chloride', 'calcium'],  # CMP
            ['bilirubin', 'albumin'],  # Liver
        ]

    used_features = set(block0_cols)
    for panel in panels:
        valid_cols = [c for c in panel if c in df.columns]
        ordered_names.extend(valid_cols)
        used_features.update(valid_cols)

    # Add orphans (unmapped features)
    test_cols = [c for c in df.columns if c not in block0_cols]
    orphans = [c for c in test_cols if c not in used_features]
    ordered_names.extend(orphans)

    return ordered_names


def visualize_policy(agent, feature_names=None, action_names=None, filename="auto_ddt_visualization", view=True):
    """Render DDT policy tree using Graphviz."""
    if not hasattr(agent.model.policy, "ddt_actor"):
        print("Error: Loaded policy is not a DDT policy.")
        return

    ddt = agent.model.policy.ddt_actor
    dot = Digraph(comment="Differentiable Decision Tree Policy")

    dot.attr(rankdir="TB", splines="line", nodesep="0.25", ranksep="0.5", margin="0")
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Arial",
        fontsize="22",
        penwidth="1.8",
        width="1.4",
        height="0.7",
        margin="0.08",
        fixedsize="false",
    )
    dot.attr("edge", fontname="Arial", fontsize="18", penwidth="1.5", arrowsize="1.2")

    queue = [(0, 0, None, None)]

    while queue:
        depth, idx, parent_name, edge_label = queue.pop(0)
        node_id = f"d{depth}_n{idx}"

        if depth == ddt.depth:
            weights = ddt.leaf_weights[idx].detach().cpu().numpy()
            best_action = int(np.argmax(weights))
            action_text = str(best_action)
            if action_names and best_action < len(action_names):
                action_text = action_names[best_action]
                if len(action_text) > 15:
                    action_text = action_text[:12] + "..."
            dot.node(node_id, action_text, shape="oval", fillcolor="#4472C4", fontcolor="white", color="#2F5597")
        else:
            layer = ddt.split_layers[depth]
            w = layer.weight[idx].detach().cpu().numpy()
            b = float(layer.bias[idx].detach().cpu().item())

            feature_idx = int(np.argmax(np.abs(w)))
            feature_val = float(w[feature_idx])
            threshold = 0.0 if abs(feature_val) < 1e-6 else (-b / feature_val)
            relation = ">" if feature_val > 0 else "<"

            feature_label = f"F_{feature_idx}"
            if feature_names and feature_idx < len(feature_names):
                feature_label = feature_names[feature_idx]
            if len(feature_label) > 12:
                feature_label = feature_label[:10] + ".."

            dot.node(node_id, f"{feature_label}\n{relation} {threshold:.2f}?", fillcolor="#FFD966", color="#BF9000")
            queue.append((depth + 1, idx * 2, node_id, "Yes"))
            queue.append((depth + 1, idx * 2 + 1, node_id, "No"))

        if parent_name is not None:
            color = "#548235" if edge_label == "Yes" else "#C00000"
            dot.edge(parent_name, node_id, label=edge_label, color=color, fontcolor=color)

    output_path = dot.render(filename, format="png", renderer="cairo", formatter="cairo")
    print(f"Visualization saved to: {output_path}")
    if view:
        dot.view()


def get_dataset_data(dataset: str):
    """Dispatch to dataset-specific preprocessing loader."""
    if dataset == "sepsis":
        return sepsis_data()
    if dataset == "aki":
        return aki_data()
    if dataset == "ferritin":
        return ferritin_data()
    raise ValueError(f"Unsupported dataset: {dataset}")


def initialize_imputer_for_cache(data_loader: Data_Loader, imputer: Imputer) -> None:
    """
    Prepare imputer internal flow model before RL builds cached imputations.

    RL cache building calls `transform_batch_same_mask`, which requires `imputer.nfm`.
    Calling `set_dataset` is sufficient to initialize this internal state.
    """
    augment_train, augment_train_mask, _ = data_loader.random_augment(data_loader.train, M=1)
    augment_train = np.nan_to_num(augment_train, nan=0.0)
    imputer.set_dataset(augment_train[:, :-1], augment_train_mask)


def run_auto_explain(dataset: str) -> None:
    """End-to-end flow: select ratio, pick run, load policy, print and visualize tree."""
    ratio = select_best_common_ratio(dataset)
    run_id, policy_path = select_best_ddt_run_and_policy(dataset, ratio)

    print(f"\nSelected DDT run_id: {run_id}")
    print(f"Selected policy path: {policy_path}")

    print("\nLoading dataset and building RL objects...")
    data, block, cost = get_dataset_data(dataset)

    dim = data.shape[1] - 1
    num_class = 2

    data_loader = Data_Loader(data, block)
    imputer = Imputer(dim, {"batch_size": 256, "lr": 1e-3, "alpha": 1e6})
    initialize_imputer_for_cache(data_loader, imputer)
    classifier = Classifier(
        dim,
        num_class,
        {
            "hidden_size": 64,
            "lr": 1e-3,
            "batch_size": 256,
            "class_weights": [1.0, 1.0],
            "save_dir": "./results",
        },
    )

    ratio_for_rl = float(ratio)
    rl_para = {
        "lr": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "penalty_ratio": ratio_for_rl,
        "ddt_depth": 4,
    }
    agent = RL(data_loader, imputer, classifier, cost, rl_para)

    print("\nLoading trained DDT policy...")
    agent.model = MaskablePPO.load(
        policy_path,
        env=agent.env,
        custom_objects={"policy_class": DDTMaskableActorCriticPolicy},
    )

    feature_names = get_actual_feature_names(dataset)
    if not feature_names:
        print("Warning: Using generic feature names fallback.")
        feature_names = build_feature_names(dim)
    
    # Prepend RL state features (classifier probabilities)
    feature_names_rl = ["CLF_Prob_Healthy", "CLF_Prob_Sick"] + feature_names + [f"Mask_{x}" for x in feature_names]
    
    num_test_blocks = len(block) - 1
    action_names = [f"Order_Block_{i}" for i in range(1, num_test_blocks + 1)] + [
        "PREDICT_HEALTHY",
        "PREDICT_SICK",
    ]

    print("\nText tree structure:")
    agent.explain_policy(feature_names=feature_names_rl, action_names=action_names)

    tree_dir = REPO_ROOT / "visualizations/tree_diag" / dataset
    tree_dir.mkdir(parents=True, exist_ok=True)
    output_name = tree_dir / f"ddt_tree_{dataset}_ratio_{_ratio_folder_name(ratio)}_run_{run_id}"
    visualize_policy(
        agent,
        feature_names=feature_names_rl,
        action_names=action_names,
        filename=str(output_name),
        view=True,
    )


def main() -> None:
    dataset = ask_dataset()
    run_auto_explain(dataset)


if __name__ == "__main__":
    main()
