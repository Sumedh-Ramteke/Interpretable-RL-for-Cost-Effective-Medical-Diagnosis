import os
import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm
import csv

from data_preprocessing.data_loader import Data_Loader
from imputer.imputation import Imputer
from classifiers.classifier_ddt import Classifier, clf_data
from rl_agent.rl_ddt import RL
from data_preprocessing.blood_panel_data_preprocessing import sepsis_data

from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

def save_log(filepath, text):
    with open(filepath, 'a') as f:
        f.write(text + "\n")
    print(text)

def main(args):
    # Unpack args
    # We ignore the passed-in params for the sweep, but keep save_dir
    _, _, _, _, save_dir = args
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 'final_results_sweep.txt')
    
    # Initialize Log
    with open(log_file, 'w') as f: f.write("=== SM-DDPO Pareto Sweep Log ===\n")

    # 1. Load Data
    data, block, cost = sepsis_data()
    DIM = data.shape[1] - 1
    NUM_CLASS = 2
    
    data_loader = Data_Loader(data, block, test_ratio = 0.2, val_ratio = 0.2)
    save_log(log_file, f"Dataset Loaded. Dim: {DIM}")

    # 2. Pretrain Imputer
    print("Pretraining Imputer...")
    imputer_para = {'batch_size': 256, 'lr': 1e-3, 'alpha': 1e6}
    imputer = Imputer(DIM, imputer_para)
    
    # Augment data for robust imputation training
    # M=5 generates 5 random masks per patient to teach imputer correlations
    augment_train, augment_train_mask, _ = data_loader.random_augment(data_loader.train, M=5)
    imputer.set_dataset(augment_train[:, :-1], augment_train_mask)
    imputer.train_model(data=None, max_iter=500)
    imputer.model_save('pretrained_imputer', save_dir=save_dir)

    # --- PARETO SWEEP CONFIGURATION ---
    # We sweep these ratios to generate the Pareto Front (Cost vs Accuracy)
    penalty_ratios = [0.5, 1, 2, 5, 10, 20]
    results_summary = []

    for ratio in penalty_ratios:
        save_log(log_file, f"\n\n=== Starting Run for Penalty Ratio: {ratio} ===")
        
        # A. Reset Classifier (Fresh start for each ratio)
        # Calculate pos_weight for class imbalance to help Classifier
        y = data[:, -1]
        n_neg = np.sum(y==0)
        n_pos = np.sum(y==1)
        pos_weight = (n_neg / n_pos) * 1.5 if n_pos > 0 else 1.0
        
        clf_para = {
            'hidden_size': 64, 
            'lr': 1e-3, 
            'batch_size': 256,
            'class_weights': [1.0, pos_weight],
            'save_dir': save_dir,
            'tree_depth': 4
        }
        classifier = Classifier(DIM, NUM_CLASS, clf_para)

        # B. Setup RL Agent
        # 'wrong_prediction_penalty' is the base cost of an error (e.g. 100)
        # 'penalty_ratio' scales that cost for False Negatives (Missing a sick patient)
        rl_para = {
            'lr': 3e-4,
            'n_steps': 2048,
            'batch_size': 256,
            'net_size': (128, 128), # Larger net for better policy stability
            'penalty_ratio': ratio,
            'wrong_prediction_penalty': 100 
        }
        
        rl_agent = RL(data_loader, imputer, classifier, cost, rl_para)
        
        # --- SEMI-MODEL-BASED OUTER LOOP ---
        # 10 loops allows the classifier to adapt to the RL agent's exploration behavior
        n_outer_loops = 10
        
        for i in range(n_outer_loops):
            save_log(log_file, f"  Outer Loop {i+1}/{n_outer_loops}")

            # 1. Train RL Policy
            # Longer training in the first loop to get off the ground
            timesteps = 50000 if i == 0 else 30000
            rl_agent.train_model(imputer, classifier, timesteps)
            
            # 2. Collect Data (Semi-Model-Based)
            # Run the policy to see what states it actually visits (Hard Data)
            new_X, new_y = rl_agent.collect_states_for_classifier(sample_size=3000)
            
            # 3. Fine-tune Classifier
            # Use collected data + validation data to ensure robustness
            # Note: Imputer returns Tensor, convert to Numpy
            val_X = imputer.transform(data_loader.val_rl[:, :-1]).detach().cpu().numpy()
            val_y = data_loader.val_rl[:, -1:]
            
            # [FIX] Concatenate Features (X) and Labels (y) horizontally
            # The Classifier expects the last column to be the label
            new_dataset = np.hstack([new_X, new_y])
            val_dataset = np.hstack([val_X, val_y])
            
            # Combine new collected data with validation data for training
            train_dataset = np.vstack([new_dataset, val_dataset])
            
            # Re-train classifier on this new mix
            # We use val_dataset for both validation and test args during this internal loop
            classifier.set_dataset(train_dataset, val_dataset, val_dataset) 
            classifier.train_model(classifier.train_dl, epochs=10, verbose=0, fresh=False)
            
            # 4. Update Agent with improved Classifier
            rl_agent.update(imputer, classifier)
            
            # Intermediate Check
            reward, _, cost_val, val_metrics, _ = rl_agent.test_model_zero_start('val', max_episodes=200)
            auroc_val = val_metrics['auroc']
            f1_val = val_metrics['f1']
            save_log(log_file, f"    Val Stats -> AUROC: {auroc_val:.4f}, F1: {f1_val:.4f}, Cost: ${cost_val:.2f}")

        # --- FINAL EVALUATION FOR THIS RATIO ---
        save_log(log_file, f"  Final Test for Ratio {ratio}...")
        reward, n_test, cost_final, metrics, dist = rl_agent.test_model_zero_start('test')
        
        res_entry = {
            'ratio': ratio,
            'auroc': metrics['auroc'],
            'f1': metrics['f1'],
            'acc': metrics['acc'],
            'cost': cost_final,
            'tests': n_test,
            'dist': dist
        }
        results_summary.append(res_entry)
        
        save_log(log_file, f"  FINAL TEST -> AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}, Acc: {metrics['acc']:.4f}, Cost: ${cost_final:.2f}, Avg Tests: {n_test:.2f}")
        rl_agent.model_save(f'policy_ratio_{ratio}', save_dir=save_dir)

    # --- SUMMARY ---
    save_log(log_file, "\n\n=== PARETO SWEEP SUMMARY ===")
    save_log(log_file, f"{'Ratio':<8} | {'AUROC':<10} | {'F1':<10} | {'Acc':<10} | {'Cost':<10} | {'Tests':<10}")
    for r in results_summary:
        save_log(log_file, f"{r['ratio']:<8} | {r['auroc']:<10.4f} | {r['f1']:<10.4f} | {r['acc']:<10.4f} | ${r['cost']:<9.2f} | {r['tests']:<10.2f}")

    print(f"Sweep Complete. Results in {log_file}")
    return

if __name__ == "__main__":
    # Default arguments to match original signature
    # (imputer_para, clf_para, rl_para, training_para, save_dir)
    save_dir = './training_results/results_ddt'
    args = ({}, {}, {}, {}, save_dir)
    main(args)