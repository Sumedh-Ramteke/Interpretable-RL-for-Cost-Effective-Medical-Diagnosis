import os
import time
import numpy as np
import pandas as pd
import torch as th

from data_preprocessing.data_loader import Data_Loader
from imputer.imputation import Imputer
from classifiers.classifier import Classifier

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, confusion_matrix

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from tqdm import tqdm
from helpers.my_result_writer import MyResultWriter

# ── Performance: pre-compute on module load ──
_NP_FLOAT32 = np.float32


def build_imputer_cache(patient_data, data_loader, imputer, n_panels):
    """
    Pre-compute imputed states for ALL patients x ALL mask patterns.
    With n_panels panels there are only 2^n_panels possible mask combinations.
    Returns a tensor of shape (n_patients, 2^n_panels, dim) on CPU.

    Also returns a numpy view (.numpy()) for zero-copy access in the env.
    """
    n_patients = len(patient_data)
    dim = patient_data.shape[1]
    n_patterns = 2 ** n_panels
    blocks = data_loader.block

    cache = th.zeros(n_patients, n_patterns, dim)

    # Pre-compute block indices for vectorised masking
    block_indices = [data_loader.block[p + 1] for p in range(n_panels)]

    for pattern in range(n_patterns):
        # Determine which panels are unobserved for this bit-pattern
        unobserved_panels = [p for p in range(n_panels) if pattern & (1 << p)]

        # Vectorised masking: copy once, set NaN columns in bulk
        masked_batch = patient_data.copy()
        for p in unobserved_panels:
            masked_batch[:, block_indices[p]] = np.nan

        # Batch transform — all same mask so only ONE matrix inverse needed
        batch_size = 1024
        parts = []
        for start in range(0, n_patients, batch_size):
            end = min(start + batch_size, n_patients)
            imputed = imputer.transform_batch_same_mask(masked_batch[start:end])
            parts.append(imputed.cpu())

        cache[:, pattern, :] = th.cat(parts, dim=0)

    return cache


class RL():
    def __init__(self, data_loader, imputer, classifier, cost, rl_para, imputer_caches=None):
        '''
        rl_para:        'lr': 0.0003
                        'n_steps':  2048
                        'batch_size': 256
                        'net_size': (128, 128)
                        'penalty_ratio': 1.0 (Lambda)
                        'wrong_prediction_penalty': 100
        '''
        self.data_loader = data_loader

        self.train = self.data_loader.train_rl
        self.test = self.data_loader.test
        self.val = self.data_loader.val_rl

        self.train_size, self.total_dim = self.train.shape
        self.num_chosen_block = len(self.data_loader.block) - 1 

        self.cost = cost
        self.para = rl_para

        ## train for #iterations steps on data
        self.lr = self.para.get('lr', 3e-4)
        self.n_steps = self.para.get('n_steps', 2048)
        self.batch_size = self.para.get('batch_size', 256)
        self.net_size = self.para.get('net_size', (128, 128))
        self.penalty_ratio = self.para.get('penalty_ratio', 1.0)
        
        # This is the "Base Penalty" for being wrong (e.g. 100)
        self.wrong_prediction_penalty = self.para.get('wrong_prediction_penalty', 100)
        
        self.file_string = f'{self.lr}_{self.n_steps}_{self.penalty_ratio}'
        
        self.env = self.env_val = self.env_test = None
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Use pre-built imputer cache if provided, otherwise build it now.
        # The imputer is frozen during RL training, so we compute all possible
        # imputed states once (patients x 2^n_panels mask patterns).
        if imputer_caches is not None:
            self._imputer_caches = imputer_caches
        else:
            self._imputer_caches = {}
            self._build_imputer_caches(imputer)

        # Initialize Environments
        self.update(imputer, classifier)

        policy_kwargs = dict(activation_fn=th.nn.SiLU, net_arch=[dict(pi=list(self.net_size), vf=list(self.net_size))])

        self.model = MaskablePPO(
            MaskableActorCriticPolicy, 
            self.env, 
            learning_rate=self.lr, 
            n_steps=self.n_steps, 
            batch_size=self.batch_size, 
            policy_kwargs=policy_kwargs, 
            verbose=0,
            device=self.device
        )
        return

    def _build_imputer_caches(self, imputer):
        """Pre-compute imputed states for all patients and all mask patterns."""
        t0 = time.time()
        splits = {
            'train': self.data_loader.train_rl[:, :-1],
            'val':   self.data_loader.val_rl[:, :-1],
            'test':  self.data_loader.test[:, :-1],
        }
        for split_name, patient_data in splits.items():
            self._imputer_caches[split_name] = build_imputer_cache(
                patient_data, self.data_loader, imputer, self.num_chosen_block
            )
        elapsed = time.time() - t0
        n_patterns = 2 ** self.num_chosen_block
        print(f"Imputer cache built in {elapsed:.1f}s "
              f"(train={splits['train'].shape[0]}, val={splits['val'].shape[0]}, "
              f"test={splits['test'].shape[0]}, patterns={n_patterns})")

    def update(self, imputer, classifier):
        self.imputer = imputer
        self.classifier = classifier

        # Fast path: if envs already exist, just swap references (avoid full reconstruction)
        if self.env is not None:
            self.env.results_writer.clear()
            self.env.imputer = imputer
            self.env.classifier = classifier
            self.env_val.results_writer.clear()
            self.env_val.imputer = imputer
            self.env_val.classifier = classifier
            self.env_test.results_writer.clear()
            self.env_test.imputer = imputer
            self.env_test.classifier = classifier
        else:
            log_dir_train = './rl_model/tmp/train_' + self.file_string + '/'
            os.makedirs(log_dir_train, exist_ok=True)
            
            # Pass the pre-computed imputer cache to each environment
            self.env = dynamic_testing('train', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty, imputer_cache=self._imputer_caches.get('train'))
            self.env_val = dynamic_testing('val', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty, imputer_cache=self._imputer_caches.get('val'))
            self.env_test = dynamic_testing('test', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty, imputer_cache=self._imputer_caches.get('test'))
        
        if hasattr(self, 'model') and self.model is not None:
            self.model.set_env(self.env)
        return

    def train_model(self, imputer, classifier, total_timesteps):
        # update the imputer and classifier
        self.update(imputer, classifier)
        self.model.learn(total_timesteps, callback=None)
        return

    def collect_states_for_classifier(self, sample_size=3000):
        """
        Runs the policy to collect partial states.
        Uses pre-allocated arrays and cached numpy state for speed.
        """
        dim = self.env.dim
        X_arr = np.empty((sample_size, dim), dtype=_NP_FLOAT32)
        y_arr = np.empty(sample_size, dtype=_NP_FLOAT32)
        obs, _ = self.env.reset()
        idx = 0

        pbar = tqdm(total=sample_size, desc="Collecting States")
        while idx < sample_size:
            action_masks = self.env.action_masks()
            action, _ = self.model.predict(obs, deterministic=False, action_masks=action_masks)
            obs, _, done, _, _ = self.env.step(action.item())

            # Use cached numpy state from env (avoids extra .detach().cpu().numpy())
            X_arr[idx] = self.env._state_np
            y_arr[idx] = self.env.patient_label
            idx += 1
            pbar.update(1)

            if done:
                obs, _ = self.env.reset()

        pbar.close()
        return X_arr, y_arr.reshape(-1, 1)

    def test_model_zero_start(self, data, max_episodes=None, generate_new_data=False):
        if data == 'test': env = self.env_test
        elif data == 'val': env = self.env_val
        elif data == 'train': env = self.env

        if not max_episodes:
            max_episodes = len(env.patient_data)
        else:
            max_episodes = min(max_episodes, len(env.patient_data))

        cumulative_reward = 0
        n_tested_ave = 0
        cost_tested_ave = 0
        action_taken = {}
        
        predict_probs = []
        pred_labels = []  # Store actual decisions
        true_labels = []

        for episode in range(max_episodes):
            obs, _ = env.reset()

            # mask out all blocks except for the first one
            mask_blocks = list(range(1, env.n_panel + 1))
            env.raw_state = env.data_loader.mask_out(env.patient.copy(), mask_blocks)
            env.get_complete_state()
            env.invalid_actions = []
            obs = env.complete_state
            
            done = False
            ep_reward = 0
            ep_cost = 0
            ep_tests = 0

            while not done:
                action_masks = env.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
                action = action.item()
                
                obs, reward, done, _, info = env.step(action)
                ep_reward += reward

                if action < env.n_panel:
                    # Test Action
                    ep_tests += 1
                    ep_cost += env.cost[action]
                    action_taken[action] = action_taken.get(action, 0) + 1
                else:
                    # Prediction Action
                    # 1. Get Prob for AUROC
                    state_tensor = env.state.unsqueeze(0) if env.state.dim() == 1 else env.state
                    with th.no_grad():
                        prob = self.classifier.predict(state_tensor).detach().cpu().numpy().flatten()
                    
                    # Store prob of positive class (1)
                    p_ill = prob[1] if len(prob) > 1 else prob[0]
                    predict_probs.append(p_ill)
                    
                    # 2. Store Predicted Label for F1/Acc
                    # The action itself determines the prediction: action = n_panel (Predict 0) or n_panel+1 (Predict 1)
                    pred_label = action - env.n_panel
                    pred_labels.append(pred_label)
                    
                    true_labels.append(env.patient_label)

            cumulative_reward += ep_reward
            n_tested_ave += ep_tests
            cost_tested_ave += ep_cost

        # --- Statistics Calculation ---
        # 1. AUROC
        try:
            if len(np.unique(true_labels)) > 1:
                auroc = roc_auc_score(true_labels, predict_probs)
            else:
                auroc = 0.5
        except:
            auroc = 0.5

        # 2. Classification Metrics (F1, Acc, etc.)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        acc = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        
        try:
            tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except:
            sensitivity = 0.0
            specificity = 0.0

        # Pack metrics
        metrics = {
            'auroc': auroc,
            'f1': f1,
            'acc': acc,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

        self.n_episodes = max_episodes
        self.n_tested_ave = n_tested_ave / max_episodes
        self.cost_tested_ave = cost_tested_ave / max_episodes
        self.cumulative_reward = cumulative_reward / max_episodes
        
        # Calculate distribution
        total_actions = sum(action_taken.values())
        if total_actions > 0:
            self.tested_distribution = {k: v/max_episodes for k, v in action_taken.items()}
        else:
            self.tested_distribution = {}

        return self.cumulative_reward, self.n_tested_ave, self.cost_tested_ave, metrics, self.tested_distribution

    def model_save(self, file_name, save_dir):
        if save_dir is None: save_dir = './rl_model/'
        os.makedirs(save_dir, exist_ok=True)
        self.model.save(save_dir + '/' + file_name + '.zip')
        return

class dynamic_testing(Env):
    def __init__(self, train_test_val, data_loader, imputer, classifier, cost, penalty_ratio, wrong_prediction_penalty=100, imputer_cache=None):
        self.results_writer = MyResultWriter()
        self.data_loader = data_loader
        self.imputer = imputer
        self.classifier = classifier
        self._imputer_cache = imputer_cache   # tensor(n_patients, 2^n_panels, dim) or None
        self._current_patient_idx = 0

        self.blocks = self.data_loader.block
        self.n_panel = len(self.blocks) - 1

        if train_test_val == 'train':
            self.patient_data = self.data_loader.train_rl[:, :-1]
            self.patient_data_label = self.data_loader.train_rl[:, -1]
        elif train_test_val == 'test':
            self.patient_data = self.data_loader.test[:, :-1]
            self.patient_data_label = self.data_loader.test[:, -1]
        else:
            self.patient_data = self.data_loader.val_rl[:, :-1]
            self.patient_data_label = self.data_loader.val_rl[:, -1]

        self.dim = len(self.patient_data[0])
        self.n_patient = len(self.patient_data)

        unique, counts = np.unique(self.patient_data_label, return_counts=True)
        self.n_label = 2 # Binary

        # Actions: [Panels] + [Predict 0, Predict 1]
        self.action_space = Discrete(self.n_panel + self.n_label)

        # Observation: [Classifier(2) + Imputed(Dim) + Mask(Dim)]
        obs_dim = self.n_label + self.dim * 2
        high = 5 * np.ones(obs_dim, dtype=_NP_FLOAT32)
        low = -5 * np.ones(obs_dim, dtype=_NP_FLOAT32)
        self.observation_space = Box(low, high, dtype=_NP_FLOAT32)

        self.cost = cost
        self.penalty_ratio = penalty_ratio
        self.wrong_prediction_penalty = wrong_prediction_penalty

        # ── Performance: pre-allocate reusable buffers ──
        self._obs_buf = np.empty(obs_dim, dtype=_NP_FLOAT32)  # reusable observation
        self._state_np = np.empty(self.dim, dtype=_NP_FLOAT32) # cached numpy state
        self._mask_buf = np.empty(self.dim, dtype=_NP_FLOAT32) # mask indicator
        self._action_mask = [True] * (self.n_panel + self.n_label)  # reusable
        # Pre-compute first feature index of each panel for fast pattern detection
        self._panel_first_idx = [self.blocks[p + 1][0] for p in range(self.n_panel)]
        self._panel_bits = [1 << p for p in range(self.n_panel)]
        # Pre-build numpy cache view if cache is available (avoids repeated .numpy())
        if self._imputer_cache is not None:
            if isinstance(self._imputer_cache, np.ndarray):
                self._imputer_cache_np = self._imputer_cache
            else:
                self._imputer_cache_np = self._imputer_cache.numpy()
        else:
            self._imputer_cache_np = None

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)

        index = np.random.randint(self.n_patient)
        self._current_patient_idx = index
        self.patient = self.patient_data[index]
        self.patient_label = self.patient_data_label[index]

        # Random start for exploration — inline to avoid list comp overhead
        n_obs_test = np.random.randint(self.n_panel + 1)
        threshold = n_obs_test / self.n_panel if self.n_panel > 0 else 0
        rands = np.random.rand(self.n_panel)
        self.unobserved_panels = [i + 1 for i in range(self.n_panel) if rands[i] > threshold]

        self.raw_state = self.data_loader.mask_out(self.patient, self.unobserved_panels)
        self.invalid_actions = [i for i in range(self.n_panel) if i + 1 not in self.unobserved_panels]

        self.get_complete_state()
        return self.complete_state, {}

    def action_masks(self):
        mask = self._action_mask
        for i in range(self.n_panel + self.n_label):
            mask[i] = True
        for act in self.invalid_actions:
            mask[act] = False
        return mask

    def _get_mask_pattern(self):
        """Derive the mask bit-pattern from self.raw_state using pre-computed indices."""
        pattern = 0
        raw = self.raw_state
        for p in range(self.n_panel):
            if np.isnan(raw[self._panel_first_idx[p]]):
                pattern |= self._panel_bits[p]
        return pattern

    def get_complete_state(self):
        # 1. Impute — use cache if available, otherwise compute on the fly
        if self._imputer_cache_np is not None:
            pattern = self._get_mask_pattern()
            # Direct numpy lookup — no tensor clone or .numpy() call
            state_np = self._imputer_cache_np[self._current_patient_idx, pattern]
            self._state_np[:] = state_np
            self.state = th.from_numpy(state_np)  # zero-copy view
        else:
            self.state = self.imputer.transform(self.raw_state)
            state_np = self.state.detach().cpu().numpy().flatten()
            self._state_np[:] = state_np

        # 2. Classify — single unsqueeze + no_grad, write directly to buffer
        with th.no_grad():
            state_input = self.state.unsqueeze(0) if self.state.dim() == 1 else self.state
            cls_np = self.classifier.predict(state_input).detach().cpu().numpy().ravel()

        # 3. Mask indicator (1 = observed, 0 = missing)
        np.isnan(self.raw_state, out=self._mask_buf)
        # _mask_buf currently has 1 for nan; flip to 1-nan
        np.subtract(1.0, self._mask_buf, out=self._mask_buf)

        # 4. Write into pre-allocated observation buffer (avoids np.concatenate alloc)
        buf = self._obs_buf
        n_cls = len(cls_np)
        buf[:n_cls] = cls_np
        buf[n_cls:n_cls + self.dim] = self._state_np
        buf[n_cls + self.dim:] = self._mask_buf

        self.complete_state = buf

    def step(self, action):
        reward = 0.0
        done = False
        
        # --- TEST ACTION ---
        if action < self.n_panel:
            self.invalid_actions.append(action)
            self.data_loader.mask_in(self.raw_state, self.patient, [action + 1])
            self.get_complete_state()

            # REWARD: Strictly subtract dollar cost
            reward = -float(self.cost[action])

        # --- PREDICTION ACTION ---
        else:
            pred = action - self.n_panel
            done = True
            
            # REWARD: Penalty Logic
            if pred == self.patient_label:
                reward = 0.0 # Correct
            else:
                if self.patient_label == 1: 
                    # False Negative (Missed Diagnosis): High Penalty * Ratio
                    reward = -self.wrong_prediction_penalty * self.penalty_ratio
                else:
                    # False Positive: Base Penalty
                    reward = -self.wrong_prediction_penalty

        # Lightweight info — avoid copying numpy array every step
        self.results_writer.write_row(None)

        return self.complete_state, reward, done, False, {}

    def render(self):
        pass