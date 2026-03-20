import os
import sys
import numpy as np
import pandas as pd
import torch as th
import random
import csv
from tqdm import tqdm

# Enable CUDA optimizations
if th.cuda.is_available():
    th.backends.cudnn.benchmark = True
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    print(f"CUDA enabled: {th.cuda.get_device_name(0)}")

# Import your existing modules
from data_preprocessing.data_loader import Data_Loader
from imputer.imputation import Imputer
from classifiers.classifier_ddt import Classifier
from rl_agent.rl_ddt import RL
from rl_agent.rl import build_imputer_cache
from data_preprocessing.blood_panel_data_preprocessing import sepsis_data, aki_data, ferritin_data
import imputer.flow_models as flow_models
import imputer.nflow as nf
# torch.multiprocessing removed — sequential execution avoids CUDA segfaults on single GPU

# --- CONFIGURATION ---
def _env_int(name, default):
    val = os.getenv(name)
    return int(val) if val is not None and val != '' else default


def _env_float(name, default):
    val = os.getenv(name)
    return float(val) if val is not None and val != '' else default


def _env_float_list(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    return [float(x.strip()) for x in raw.split(',') if x.strip()]


NUM_RUNS = _env_int('NUM_RUNS', 10)
PENALTY_RATIOS = _env_float_list('PENALTY_RATIOS', [1, 2, 3, 5, 10])

# F1-focused but architecture-preserving defaults
POS_WEIGHT_SCALE = _env_float('POS_WEIGHT_SCALE', 1.8)
CLF_LR = _env_float('CLF_LR', 8e-4)
CLF_BATCH_SIZE = _env_int('CLF_BATCH_SIZE', 256)
CLF_FINETUNE_EPOCHS = _env_int('CLF_FINETUNE_EPOCHS', 12)
TREE_DEPTH = _env_int('TREE_DEPTH', 4)

RL_LR = _env_float('RL_LR', 2e-4)
RL_N_STEPS = _env_int('RL_N_STEPS', 2048)
RL_BATCH_SIZE = _env_int('RL_BATCH_SIZE', 256)
RL_GAMMA = _env_float('RL_GAMMA', 0.995)
RL_GAE_LAMBDA = _env_float('RL_GAE_LAMBDA', 0.95)
RL_ENT_COEF = _env_float('RL_ENT_COEF', 0.005)
RL_CLIP_RANGE = _env_float('RL_CLIP_RANGE', 0.2)
WRONG_PREDICTION_PENALTY = _env_float('WRONG_PREDICTION_PENALTY', 120)

N_OUTER_LOOPS = _env_int('N_OUTER_LOOPS', 10)
FIRST_LOOP_TIMESTEPS = _env_int('FIRST_LOOP_TIMESTEPS', 50000)
SUBSEQUENT_LOOP_TIMESTEPS = _env_int('SUBSEQUENT_LOOP_TIMESTEPS', 30000)
STATE_SAMPLE_SIZE = _env_int('STATE_SAMPLE_SIZE', 4000)
IMPUTER_MAX_ITER = _env_int('IMPUTER_MAX_ITER', 500)
IMPUTER_AUGMENT_M = _env_int('IMPUTER_AUGMENT_M', 5)

DATASET_OPTIONS = {
    '1': ('sepsis',   sepsis_data),
    '2': ('aki',      aki_data),
    '3': ('ferritin', ferritin_data),
}


def select_dataset():
    env_choice = os.getenv('DATASET_CHOICE', '').strip()
    if env_choice == '4':
        print("Selected from DATASET_CHOICE: ALL DISEASES")
        return 'all', None
    if env_choice in DATASET_OPTIONS:
        mode, data_fn = DATASET_OPTIONS[env_choice]
        print(f"Selected from DATASET_CHOICE: {mode.upper()}")
        return mode, data_fn

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

# --- UTILITIES ---

def set_global_seed(seed):
    """Sets random seeds for reproducibility per run."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

class SeededImputer(Imputer):
    """
    Subclass of Imputer to allow external seeding.
    """
    def init_model(self):
        # We removed torch.manual_seed(0) to allow randomness.
        # However, to ensure stability, we rely on init_zeros=True in the MLP
        # and lower learning rates in the optimizer.
        num_flows = 32
        
        flows = []
        for i in range(num_flows):
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
            completed.add((int(row['run_id']), float(row['ratio'])))
    except Exception as e:
        print(f"Warning: Could not read existing metrics file. Starting fresh. Error: {e}")
    return completed

def run_single_ratio_experiment(run_id, ratio, data_loader, imputer_caches, precomputed_val_X, precomputed_val_y, cost, dim, num_class, save_dir):
    """Train one ratio experiment."""
    print(f"--- Run {run_id} | Ratio {ratio} | PID {os.getpid()} ---")
    
    # [FIX] Define a specific directory for this Run + Ratio combination
    # This ensures models don't overwrite each other
    current_run_dir = os.path.join(save_dir, f"run_{run_id}", f"ratio_{ratio}")
    if not os.path.exists(current_run_dir):
        os.makedirs(current_run_dir)
    
    # 1. Setup Classifier
    y = data_loader.data[:, -1]
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    pos_weight = (n_neg / n_pos) * POS_WEIGHT_SCALE if n_pos > 0 else 1.0
    
    clf_para = {
        'hidden_size': 64, 
        'lr': CLF_LR,
        'batch_size': CLF_BATCH_SIZE,
        'class_weights': [1.0, pos_weight],
        'save_dir': current_run_dir, # Pass specific dir to classifier
        'tree_depth': TREE_DEPTH
    }

    classifier = Classifier(dim, num_class, clf_para)

    # 2. Setup RL Agent
    rl_para = {
        'lr': RL_LR,
        'n_steps': RL_N_STEPS,
        'batch_size': RL_BATCH_SIZE,
        'net_size': (128, 128),
        'penalty_ratio': ratio,
        'wrong_prediction_penalty': WRONG_PREDICTION_PENALTY,
        'ddt_depth': TREE_DEPTH,
        'gamma': RL_GAMMA,
        'gae_lambda': RL_GAE_LAMBDA,
        'ent_coef': RL_ENT_COEF,
        'clip_range': RL_CLIP_RANGE
    }
    
    rl_agent = RL(data_loader, None, classifier, cost, rl_para, imputer_caches=imputer_caches)
    
    # 3. Outer Loop Training
    n_outer_loops = N_OUTER_LOOPS
    
    for i in range(n_outer_loops):
        # A. Train RL Policy
        timesteps = FIRST_LOOP_TIMESTEPS if i == 0 else SUBSEQUENT_LOOP_TIMESTEPS
        rl_agent.train_model(None, classifier, timesteps)
        
        # B. Collect Data
        new_X, new_y = rl_agent.collect_states_for_classifier(sample_size=STATE_SAMPLE_SIZE)
        
        # C. Fine-tune Classifier (use pre-computed validation data)
        new_dataset = np.hstack([new_X, new_y])
        val_dataset = np.hstack([precomputed_val_X, precomputed_val_y])
        train_dataset = np.vstack([new_dataset, val_dataset])
        
        classifier.set_dataset(train_dataset, val_dataset, val_dataset) 
        classifier.train_model(classifier.train_dl, epochs=CLF_FINETUNE_EPOCHS, verbose=0, fresh=False)
        
        # D. Update Agent
        rl_agent.update(None, classifier)

    # 4. Final Evaluation
    reward, n_test, cost_final, metrics, dist = rl_agent.test_model_zero_start('test')
    
    # --- [FIX] SAVE THE MODELS ---
    print(f"Saving models to: {current_run_dir}")
    try:
        # Save the RL Agent (Policy)
        rl_agent.model_save(f'policy_ratio_{ratio}', save_dir=current_run_dir)
        
        # (Optional) Save the Classifier if you want to inspect decision boundaries later
        classifier.model_save(f'classifier_ratio_{ratio}', save_dir=current_run_dir)
        
        # Verify file creation
        expected_file = os.path.join(current_run_dir, f'policy_ratio_{ratio}.zip')
        if os.path.exists(expected_file):
            print(f" [OK] Saved {expected_file}")
        else:
            print(f" [WARNING] Model file not found at {expected_file}")
            
    except Exception as e:
        print(f" [ERROR] Failed to save model: {e}")
    # -----------------------------
    
    return {
        'run_id': run_id,
        'ratio': ratio,
        'auroc': metrics['auroc'],
        'f1': metrics['f1'],
        'acc': metrics['acc'],
        'precision': metrics.get('precision', 0),
        'sensitivity': metrics.get('sensitivity', 0),
        'specificity': metrics.get('specificity', 0),
        'cost': cost_final,
        'avg_tests': n_test,
        'reward': reward
    }

def main():
    # --- Dataset Selection ---
    mode, data_fn = select_dataset()
    if mode == 'all':
        selected_datasets = [
            ('sepsis', sepsis_data),
            ('aki', aki_data),
            ('ferritin', ferritin_data),
        ]
    else:
        selected_datasets = [(mode, data_fn)]

    for mode, data_fn in selected_datasets:
        SAVE_DIR = f'./training_results/results_multi_run_ddt/{mode}'
        METRICS_FILE = os.path.join(SAVE_DIR, 'all_runs_metrics.csv')

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        fieldnames = ['run_id', 'ratio', 'auroc', 'f1', 'acc', 'precision',
                      'sensitivity', 'specificity', 'cost', 'avg_tests', 'reward']

        if not os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        # 1. Main Training Loop
        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n{'='*40}\n[{mode.upper()}] STARTING RUN {run_id}/{NUM_RUNS}\n{'='*40}")

            completed_tasks = get_completed_tasks(METRICS_FILE)

            ratios_needed = [r for r in PENALTY_RATIOS if (run_id, r) not in completed_tasks]
            if not ratios_needed:
                print(f"Run {run_id} already completed. Skipping.")
                continue

            # A. Set Global Seed
            set_global_seed(run_id * 100)

            # B. Load Data
            data, block, cost_raw = data_fn()
            dim = data.shape[1] - 1
            num_class = 2
            data_loader = Data_Loader(data, block, test_ratio=0.2, val_ratio=0.2)

            # C. Pretrain Imputer
            print(f"Run {run_id}: Pretraining Imputer...")

            # FIX 1: Lower learning rate to prevent divergence on random initialization
            imputer_para = {'batch_size': 256, 'lr': 1e-4, 'alpha': 1e6}

            imputer = SeededImputer(dim, imputer_para)

            # Generate augmented training data
            augment_train, augment_train_mask, _ = data_loader.random_augment(data_loader.train, M=IMPUTER_AUGMENT_M)

            # FIX 2: Sanitize input data. Replace NaNs with 0.0 BEFORE passing to Imputer.
            # The imputer uses 'augment_train_mask' to know what is missing.
            # Passing NaNs directly into a Neural Network (even if masked later) is dangerous.
            augment_train = np.nan_to_num(augment_train, nan=0.0)

            imputer.set_dataset(augment_train[:, :-1], augment_train_mask)

            try:
                imputer.train_model(data=None, max_iter=IMPUTER_MAX_ITER)

                # [FIX] Save the Imputer as well (optional, but good for reproducibility)
                imputer_save_dir = os.path.join(SAVE_DIR, f"run_{run_id}")
                if not os.path.exists(imputer_save_dir):
                    os.makedirs(imputer_save_dir)
                imputer.model_save('pretrained_imputer', save_dir=imputer_save_dir)

            except ValueError as e:
                print(f"CRITICAL ERROR in Run {run_id} Imputer Training: {e}")
                print("Retrying with a fallback seed/configuration or skipping...")
                # Optional: You could add retry logic here, but fixing LR and NaN usually solves it.
                raise e

            # D. Pre-build imputer cache & validation data (shared across all ratios)
            print(f"Run {run_id}: Building imputer cache...")
            n_panels = len(block) - 1
            # Convert caches to numpy immediately to avoid torch tensor pickling issues
            imputer_caches = {
                'train': build_imputer_cache(data_loader.train_rl[:, :-1], data_loader, imputer, n_panels).numpy(),
                'val': build_imputer_cache(data_loader.val_rl[:, :-1], data_loader, imputer, n_panels).numpy(),
                'test': build_imputer_cache(data_loader.test[:, :-1], data_loader, imputer, n_panels).numpy(),
            }
            val_X_raw = data_loader.val_rl[:, :-1]
            val_X_clean = np.nan_to_num(val_X_raw, nan=0.0)
            precomputed_val_X = imputer.transform(th.tensor(val_X_clean, dtype=th.float32)).detach().cpu().numpy()
            precomputed_val_y = data_loader.val_rl[:, -1:]
            print(f"Run {run_id}: Cache built. Training {len(ratios_needed)} ratios sequentially...")

            # E. Train ratios sequentially (single GPU cannot benefit from parallel CUDA contexts)
            results = []
            for ratio in ratios_needed:
                result = run_single_ratio_experiment(
                    run_id, ratio, data_loader, imputer_caches, precomputed_val_X,
                    precomputed_val_y, cost_raw, dim, num_class, SAVE_DIR
                )
                results.append(result)

            for result_row in results:
                with open(METRICS_FILE, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result_row)
                print(f"Saved results for Run {run_id}, Ratio {result_row['ratio']}")

            # Free GPU memory from imputer before next run
            del imputer, imputer_caches
            th.cuda.empty_cache()

        # 2. Aggregation
        print(f"\n{'='*40}\n[{mode.upper()}] COMPUTING AGGREGATE RESULTS\n{'='*40}")
        df = pd.read_csv(METRICS_FILE)
        summary = df.groupby('ratio').agg(['mean', 'std'])
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary_path = os.path.join(SAVE_DIR, 'summary_results.csv')
        summary.to_csv(summary_path)
        print("Aggregate Results (Means):")
        print(summary[['auroc_mean', 'f1_mean', 'cost_mean', 'avg_tests_mean']])

if __name__ == "__main__":
    main()