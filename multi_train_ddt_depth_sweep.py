import os
import numpy as np
import pandas as pd
import torch as th
import random
import csv

# Enable CUDA optimizations
if th.cuda.is_available():
    th.backends.cudnn.benchmark = True
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    print(f"CUDA enabled: {th.cuda.get_device_name(0)}")

from data_preprocessing.data_loader import Data_Loader
from imputer.imputation import Imputer
from classifiers.classifier_ddt import Classifier
from rl_agent.rl_ddt import RL
from rl_agent.rl import build_imputer_cache
from data_preprocessing.blood_panel_data_preprocessing import sepsis_data, aki_data, ferritin_data
import imputer.flow_models as flow_models
import imputer.nflow as nf


# --- CONFIGURATION ---
NUM_RUNS = 10
PENALTY_RATIOS = [1, 2, 3, 5, 10]
DEFAULT_TREE_DEPTHS = [2, 3, 4, 5]

DATASET_OPTIONS = {
    '1': ('sepsis', sepsis_data),
    '2': ('aki', aki_data),
    '3': ('ferritin', ferritin_data),
}


def select_dataset():
    print("\nSelect dataset to train on:")
    print("  1. Sepsis")
    print("  2. AKI")
    print("  3. Ferritin")
    print("  4. All")
    while True:
        choice = input("Enter choice [1/2/3/4]: ").strip()
        if choice == '4':
            print("Selected: ALL DISEASES")
            return 'all', None
        if choice in DATASET_OPTIONS:
            mode, data_fn = DATASET_OPTIONS[choice]
            print(f"Selected: {mode.upper()}")
            return mode, data_fn
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


def select_tree_depths():
    default_text = ",".join(str(depth) for depth in DEFAULT_TREE_DEPTHS)
    print("\nSelect tree depths to evaluate.")
    print(f"Press Enter to use defaults: {default_text}")

    while True:
        raw_value = input("Enter comma-separated depths: ").strip()
        if not raw_value:
            print(f"Selected default depths: {DEFAULT_TREE_DEPTHS}")
            return DEFAULT_TREE_DEPTHS

        try:
            depths = sorted({int(item.strip()) for item in raw_value.split(',') if item.strip()})
        except ValueError:
            print("Invalid input. Enter integers separated by commas, e.g. 2,3,4,5.")
            continue

        if not depths or any(depth < 1 for depth in depths):
            print("Depth values must be positive integers.")
            continue

        print(f"Selected depths: {depths}")
        return depths


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


class SeededImputer(Imputer):
    def init_model(self):
        num_flows = 32

        flows = []
        for _ in range(num_flows):
            if self.d % 2 == 1:
                param_map = flow_models.MLP([self.d // 2 + 1, 32, 32, self.d], init_zeros=True)
            else:
                param_map = flow_models.MLP([self.d // 2, 32, 32, self.d], init_zeros=True)
            flows.append(nf.AffineCouplingBlock(param_map))
            flows.append(nf.Permute(self.d, mode='swap'))

        self.nfm = nf.NormalizingFlow(q0=None, flows=flows).to(self.device)


def get_completed_tasks(csv_path):
    completed = set()
    if not os.path.exists(csv_path):
        return completed

    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            completed.add((int(row['run_id']), int(row['tree_depth']), float(row['ratio'])))
    except Exception as error:
        print(f"Warning: Could not read existing metrics file. Starting fresh. Error: {error}")
    return completed


def run_single_ratio_experiment(
    run_id,
    tree_depth,
    ratio,
    data_loader,
    imputer_caches,
    precomputed_val_X,
    precomputed_val_y,
    cost,
    dim,
    num_class,
    save_dir,
):
    print(f"--- Run {run_id} | Depth {tree_depth} | Ratio {ratio} | PID {os.getpid()} ---")

    current_run_dir = os.path.join(save_dir, f"run_{run_id}", f"depth_{tree_depth}", f"ratio_{ratio}")
    if not os.path.exists(current_run_dir):
        os.makedirs(current_run_dir)

    y = data_loader.data[:, -1]
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    pos_weight = (n_neg / n_pos) * 1.5 if n_pos > 0 else 1.0

    clf_para = {
        'hidden_size': 64,
        'lr': 1e-3,
        'batch_size': 256,
        'class_weights': [1.0, pos_weight],
        'save_dir': current_run_dir,
        'tree_depth': tree_depth,
    }
    classifier = Classifier(dim, num_class, clf_para)

    rl_para = {
        'lr': 3e-4,
        'n_steps': 2048,
        'batch_size': 256,
        'net_size': (128, 128),
        'penalty_ratio': ratio,
        'wrong_prediction_penalty': 100,
        'ddt_depth': tree_depth,
    }
    rl_agent = RL(data_loader, None, classifier, cost, rl_para, imputer_caches=imputer_caches)

    for outer_loop in range(10):
        timesteps = 50000 if outer_loop == 0 else 30000
        rl_agent.train_model(None, classifier, timesteps)

        new_X, new_y = rl_agent.collect_states_for_classifier(sample_size=3000)

        new_dataset = np.hstack([new_X, new_y])
        val_dataset = np.hstack([precomputed_val_X, precomputed_val_y])
        train_dataset = np.vstack([new_dataset, val_dataset])

        classifier.set_dataset(train_dataset, val_dataset, val_dataset)
        classifier.train_model(classifier.train_dl, epochs=10, verbose=0, fresh=False)
        rl_agent.update(None, classifier)

    reward, n_test, cost_final, metrics, _ = rl_agent.test_model_zero_start('test')

    print(f"Saving models to: {current_run_dir}")
    try:
        rl_agent.model_save(f'policy_depth_{tree_depth}_ratio_{ratio}', save_dir=current_run_dir)
        classifier.model_save(f'classifier_depth_{tree_depth}_ratio_{ratio}', save_dir=current_run_dir)
    except Exception as error:
        print(f" [ERROR] Failed to save model: {error}")

    return {
        'run_id': run_id,
        'tree_depth': tree_depth,
        'ratio': ratio,
        'auroc': metrics['auroc'],
        'f1': metrics['f1'],
        'acc': metrics['acc'],
        'precision': metrics.get('precision', 0),
        'sensitivity': metrics.get('sensitivity', 0),
        'specificity': metrics.get('specificity', 0),
        'cost': cost_final,
        'avg_tests': n_test,
        'reward': reward,
    }


def main():
    mode, data_fn = select_dataset()
    tree_depths = select_tree_depths()
    if mode == 'all':
        selected_datasets = [
            ('sepsis', sepsis_data),
            ('aki', aki_data),
            ('ferritin', ferritin_data),
        ]
    else:
        selected_datasets = [(mode, data_fn)]

    for mode, data_fn in selected_datasets:
        save_dir = f'./training_results/results_multi_run_ddt_depth/{mode}'
        metrics_file = os.path.join(save_dir, 'all_runs_metrics.csv')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fieldnames = [
            'run_id', 'tree_depth', 'ratio', 'auroc', 'f1', 'acc', 'precision',
            'sensitivity', 'specificity', 'cost', 'avg_tests', 'reward'
        ]

        if not os.path.exists(metrics_file):
            with open(metrics_file, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()

        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n{'=' * 48}\n[{mode.upper()}] STARTING RUN {run_id}/{NUM_RUNS}\n{'=' * 48}")

            completed_tasks = get_completed_tasks(metrics_file)
            experiments_needed = [
                (tree_depth, ratio)
                for tree_depth in tree_depths
                for ratio in PENALTY_RATIOS
                if (run_id, tree_depth, ratio) not in completed_tasks
            ]
            if not experiments_needed:
                print(f"Run {run_id} already completed for all selected depths and ratios. Skipping.")
                continue

            set_global_seed(run_id * 100)

            data, block, cost_raw = data_fn()
            dim = data.shape[1] - 1
            num_class = 2
            data_loader = Data_Loader(data, block, test_ratio=0.2, val_ratio=0.2)

            print(f"Run {run_id}: Pretraining Imputer...")
            imputer_para = {'batch_size': 256, 'lr': 1e-4, 'alpha': 1e6}
            imputer = SeededImputer(dim, imputer_para)

            augment_train, augment_train_mask, _ = data_loader.random_augment(data_loader.train, M=5)
            augment_train = np.nan_to_num(augment_train, nan=0.0)
            imputer.set_dataset(augment_train[:, :-1], augment_train_mask)

            try:
                imputer.train_model(data=None, max_iter=500)
                imputer_save_dir = os.path.join(save_dir, f"run_{run_id}")
                if not os.path.exists(imputer_save_dir):
                    os.makedirs(imputer_save_dir)
                imputer.model_save('pretrained_imputer', save_dir=imputer_save_dir)
            except ValueError as error:
                print(f"CRITICAL ERROR in Run {run_id} Imputer Training: {error}")
                raise error

            print(f"Run {run_id}: Building imputer cache...")
            n_panels = len(block) - 1
            imputer_caches = {
                'train': build_imputer_cache(data_loader.train_rl[:, :-1], data_loader, imputer, n_panels).numpy(),
                'val': build_imputer_cache(data_loader.val_rl[:, :-1], data_loader, imputer, n_panels).numpy(),
                'test': build_imputer_cache(data_loader.test[:, :-1], data_loader, imputer, n_panels).numpy(),
            }
            val_X_raw = data_loader.val_rl[:, :-1]
            val_X_clean = np.nan_to_num(val_X_raw, nan=0.0)
            precomputed_val_X = imputer.transform(th.tensor(val_X_clean, dtype=th.float32)).detach().cpu().numpy()
            precomputed_val_y = data_loader.val_rl[:, -1:]
            print(f"Run {run_id}: Cache built. Training {len(experiments_needed)} depth-ratio combinations sequentially...")

            results = []
            for tree_depth, ratio in experiments_needed:
                result = run_single_ratio_experiment(
                    run_id,
                    tree_depth,
                    ratio,
                    data_loader,
                    imputer_caches,
                    precomputed_val_X,
                    precomputed_val_y,
                    cost_raw,
                    dim,
                    num_class,
                    save_dir,
                )
                results.append(result)

            for result_row in results:
                with open(metrics_file, 'a', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writerow(result_row)
                print(
                    f"Saved results for Run {run_id}, Depth {result_row['tree_depth']}, "
                    f"Ratio {result_row['ratio']}"
                )

            del imputer, imputer_caches
            th.cuda.empty_cache()

        print(f"\n{'=' * 48}\n[{mode.upper()}] COMPUTING AGGREGATE RESULTS\n{'=' * 48}")
        df = pd.read_csv(metrics_file)
        summary = df.groupby(['tree_depth', 'ratio']).agg(['mean', 'std'])
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary_path = os.path.join(save_dir, 'summary_results.csv')
        summary.to_csv(summary_path)
        print("Aggregate Results (Means):")
        print(summary[['auroc_mean', 'f1_mean', 'cost_mean', 'avg_tests_mean']])


if __name__ == "__main__":
    main()
