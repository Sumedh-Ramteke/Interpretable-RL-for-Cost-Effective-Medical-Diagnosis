import os
import time
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, confusion_matrix
from tqdm import tqdm

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Assuming these exist in your project structure
from data_preprocessing.data_loader import Data_Loader
from imputer.imputation import Imputer
from classifiers.classifier_ddt import Classifier
from helpers.my_result_writer import MyResultWriter
from rl_agent.rl import build_imputer_cache

# ── Performance constant ──
_NP_FLOAT32 = np.float32

# ==========================================
# 1. DDT Implementation Classes
# ==========================================

class DifferentiableDecisionTree(nn.Module):
    """
    A Differentiable Decision Tree (DDT) module.
    Paper Ref: Eq. 4: mu_eta(x) = sigmoid(alpha * (beta * x - phi))
    """
    def __init__(self, input_dim, output_dim, depth=4):
        super(DifferentiableDecisionTree, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.num_leaves = 2 ** depth
        
        # We model the splits as Linear layers followed by Sigmoid
        self.split_layers = nn.ModuleList()
        
        # Construct layers for each depth level
        for d in range(depth):
            nodes_at_depth = 2 ** d
            layer = nn.Linear(input_dim, nodes_at_depth)
            self.split_layers.append(layer)
            
        # Leaf parameters (Action Logits or Value)
        self.leaf_weights = nn.Parameter(th.randn(self.num_leaves, output_dim))
        
        # Inverse temperature parameter
        self.alpha = nn.Parameter(th.ones(1)) 

    def forward(self, x):
        batch_size = x.size(0)
        path_probs = th.ones(batch_size, 1, device=x.device)
        
        for d in range(self.depth):
            splits = self.split_layers[d](x)
            decisions = th.sigmoid(self.alpha * splits)
            
            prob_left = path_probs * decisions
            prob_right = path_probs * (1 - decisions)
            
            path_probs = th.stack([prob_left, prob_right], dim=2).view(batch_size, -1)
            
        output = th.matmul(path_probs, self.leaf_weights)
        return output

class DDTFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int):
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)

class DDTMlpExtractor(nn.Module):
    """Dummy extractor to satisfy SB3 interface"""
    def __init__(self, features_dim):
        super().__init__()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim
        
    def forward(self, features):
        return features, features

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return features

class DDTMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, ddt_depth=4, **kwargs):
        self.ddt_depth = ddt_depth
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DDTFeaturesExtractor,
            features_extractor_kwargs={"features_dim": np.prod(observation_space.shape)},
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.ortho_init = False
        input_dim = np.prod(self.observation_space.shape)
        
        if isinstance(self.action_space, Discrete):
            action_dim = self.action_space.n
        else:
            raise NotImplementedError("DDT requires Discrete Action Space")

        self.ddt_actor = DifferentiableDecisionTree(input_dim, action_dim, self.ddt_depth)
        self.ddt_critic = DifferentiableDecisionTree(input_dim, 1, self.ddt_depth)
        self.mlp_extractor = DDTMlpExtractor(input_dim)

    def forward(self, obs, deterministic=False, action_masks=None):
        features = self.extract_features(obs)
        action_logits = self.ddt_actor(features)
        values = self.ddt_critic(features)
        distribution = self._get_action_dist_from_logits(action_logits, action_masks=action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        return actions, values, log_probs

    def _predict(self, observation, deterministic=False, action_masks=None):
        features = self.extract_features(observation)
        action_logits = self.ddt_actor(features)
        distribution = self._get_action_dist_from_logits(action_logits, action_masks)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions, action_masks=None):
        features = self.extract_features(obs)
        action_logits = self.ddt_actor(features)
        values = self.ddt_critic(features)
        distribution = self._get_action_dist_from_logits(action_logits, action_masks)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_probs, entropy

    def predict_values(self, obs):
        features = self.extract_features(obs)
        return self.ddt_critic(features)

    def _get_action_dist_from_logits(self, action_logits, action_masks=None):
        try:
            return self.action_dist.proba_distribution(action_logits, action_masks=action_masks)
        except TypeError:
            if action_masks is not None:
                if isinstance(action_masks, list):
                    action_masks = th.tensor(action_masks, device=action_logits.device, dtype=th.bool)
                elif isinstance(action_masks, np.ndarray):
                    action_masks = th.as_tensor(action_masks, device=action_logits.device, dtype=th.bool)
                else:
                    action_masks = action_masks.bool()
                
                HUGE_NEG = -1e8
                action_logits = th.where(action_masks, action_logits, th.tensor(HUGE_NEG, device=action_logits.device))
            
            distribution = self.action_dist.proba_distribution(action_logits)
            if action_masks is not None and hasattr(distribution, "action_masks"):
                distribution.action_masks = action_masks
            return distribution

# ==========================================
# 2. Main RL Class
# ==========================================

class RL():
    def __init__(self, data_loader, imputer, classifier, cost, rl_para, imputer_caches=None):
        '''
        rl_para:        'lr': 0.0003
                        'n_steps':  2048
                        'batch_size': 256
                        'penalty_ratio': 1.0 (Lambda)
                        'wrong_prediction_penalty': 100
                        'ddt_depth': 4 (Depth of the decision tree)
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
        # self.net_size = self.para.get('net_size', (128, 128)) # MLP not used
        self.penalty_ratio = self.para.get('penalty_ratio', 1.0)
        self.wrong_prediction_penalty = self.para.get('wrong_prediction_penalty', 100)
        
        # New parameter for DDT
        self.ddt_depth = self.para.get('ddt_depth', 4)
        
        self.file_string = f'{self.lr}_{self.n_steps}_{self.penalty_ratio}_depth{self.ddt_depth}'
        
        self.env = self.env_val = self.env_test = None
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Use pre-built imputer cache if provided, otherwise build it now.
        if imputer_caches is not None:
            self._imputer_caches = imputer_caches
        else:
            self._imputer_caches = {}
            self._build_imputer_caches(imputer)

        # Initialize Environments
        self.update(imputer, classifier)

        # DDT Policy Arguments
        # We pass ddt_depth to the policy init
        policy_kwargs = dict(ddt_depth=self.ddt_depth)

        # Replaced standard MaskableActorCriticPolicy with DDTMaskableActorCriticPolicy
        self.model = MaskablePPO(
            DDTMaskableActorCriticPolicy, 
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
        print(f"DDT Imputer cache built in {elapsed:.1f}s "
              f"(train={splits['train'].shape[0]}, val={splits['val'].shape[0]}, "
              f"test={splits['test'].shape[0]}, patterns={n_patterns})")

    def update(self, imputer, classifier):
        self.imputer = imputer
        self.classifier = classifier

        # Fast path: reuse envs, just swap references
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
            
            self.env = dynamic_testing('train', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty, imputer_cache=self._imputer_caches.get('train'))
            self.env_val = dynamic_testing('val', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty, imputer_cache=self._imputer_caches.get('val'))
            self.env_test = dynamic_testing('test', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty, imputer_cache=self._imputer_caches.get('test'))
        
        if hasattr(self, 'model') and self.model is not None:
            self.model.set_env(self.env)
        return

    def train_model(self, imputer, classifier, total_timesteps):
        self.update(imputer, classifier)
        self.model.learn(total_timesteps, callback=None)
        return

    def explain_policy(self, feature_names=None, action_names=None):
        """
        Prints the interpretable decision tree logic extracted from the trained DDT actor.
        """
        if not hasattr(self.model.policy, 'ddt_actor'):
            print("Policy is not a DDT.")
            return

        print("\n=== Extracted Interpretable Decision Tree ===")
        ddt = self.model.policy.ddt_actor
        
        def print_node(depth, node_idx_in_layer, prefix=""):
            if depth == ddt.depth:
                # LEAF NODE
                global_leaf_idx = node_idx_in_layer
                weights = ddt.leaf_weights[global_leaf_idx].detach().cpu().numpy()
                best_action = np.argmax(weights)
                
                act_str = str(best_action)
                if action_names and best_action < len(action_names):
                    act_str = action_names[best_action]
                
                print(f"{prefix}└── Action: {act_str} (Logits: {weights})")
                return

            # INTERNAL NODE
            layer = ddt.split_layers[depth]
            w = layer.weight[node_idx_in_layer].detach().cpu().numpy()
            b = layer.bias[node_idx_in_layer].detach().cpu().item()
            
            # Discretization
            feature_idx = np.argmax(np.abs(w))
            feature_val = w[feature_idx]
            threshold = -b / feature_val
            
            fname = f"Feature_{feature_idx}"
            if feature_names and feature_idx < len(feature_names):
                fname = feature_names[feature_idx]
            
            # Branch logic
            print(f"{prefix}├── IF {fname} > {threshold:.4f}:")
            print_node(depth + 1, node_idx_in_layer * 2, prefix + "│   ")
            
            print(f"{prefix}├── ELSE:")
            print_node(depth + 1, node_idx_in_layer * 2 + 1, prefix + "│   ")

        print_node(0, 0)
        print("=============================================\n")

    def collect_states_for_classifier(self, sample_size=3000):
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
        pred_labels = [] 
        true_labels = []

        for episode in range(max_episodes):
            obs, _ = env.reset()

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
                    ep_tests += 1
                    ep_cost += env.cost[action]
                    action_taken[action] = action_taken.get(action, 0) + 1
                else:
                    state_tensor = env.state.unsqueeze(0) if env.state.dim() == 1 else env.state
                    with th.no_grad():
                        prob = self.classifier.predict(state_tensor).detach().cpu().numpy().flatten()
                    
                    p_ill = prob[1] if len(prob) > 1 else prob[0]
                    predict_probs.append(p_ill)
                    
                    pred_label = action - env.n_panel
                    pred_labels.append(pred_label)
                    true_labels.append(env.patient_label)

            cumulative_reward += ep_reward
            n_tested_ave += ep_tests
            cost_tested_ave += ep_cost

        # --- Statistics Calculation ---
        try:
            if len(np.unique(true_labels)) > 1:
                auroc = roc_auc_score(true_labels, predict_probs)
            else:
                auroc = 0.5
        except:
            auroc = 0.5

        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        acc = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        
        try:
            tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except:
            sensitivity = 0.0
            specificity = 0.0

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
        self._imputer_cache = imputer_cache
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
        self.n_label = 2

        self.action_space = Discrete(self.n_panel + self.n_label)

        obs_dim = self.n_label + self.dim * 2
        high = 5 * np.ones(obs_dim, dtype=_NP_FLOAT32)
        low = -5 * np.ones(obs_dim, dtype=_NP_FLOAT32)
        self.observation_space = Box(low, high, dtype=_NP_FLOAT32)

        self.cost = cost
        self.penalty_ratio = penalty_ratio
        self.wrong_prediction_penalty = wrong_prediction_penalty

        # Pre-allocate buffers
        self._obs_buf = np.empty(obs_dim, dtype=_NP_FLOAT32)
        self._state_np = np.empty(self.dim, dtype=_NP_FLOAT32)
        self._mask_buf = np.empty(self.dim, dtype=_NP_FLOAT32)
        self._action_mask = [True] * (self.n_panel + self.n_label)
        self._panel_first_idx = [self.blocks[p + 1][0] for p in range(self.n_panel)]
        self._panel_bits = [1 << p for p in range(self.n_panel)]
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
        pattern = 0
        raw = self.raw_state
        for p in range(self.n_panel):
            if np.isnan(raw[self._panel_first_idx[p]]):
                pattern |= self._panel_bits[p]
        return pattern

    def get_complete_state(self):
        if self._imputer_cache_np is not None:
            pattern = self._get_mask_pattern()
            state_np = self._imputer_cache_np[self._current_patient_idx, pattern]
            self._state_np[:] = state_np
            self.state = th.from_numpy(state_np)
        else:
            self.state = self.imputer.transform(self.raw_state)
            state_np = self.state.detach().cpu().numpy().flatten()
            self._state_np[:] = state_np

        with th.no_grad():
            state_input = self.state.unsqueeze(0) if self.state.dim() == 1 else self.state
            cls_np = self.classifier.predict(state_input).detach().cpu().numpy().ravel()

        np.isnan(self.raw_state, out=self._mask_buf)
        np.subtract(1.0, self._mask_buf, out=self._mask_buf)

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

        self.results_writer.write_row(None)

        return self.complete_state, reward, done, False, {}

    def render(self):
        pass