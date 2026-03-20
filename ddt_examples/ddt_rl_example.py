import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

class DifferentiableDecisionTree(nn.Module):
    """
    A Differentiable Decision Tree (DDT) module as described in the paper.
    It uses soft splits (sigmoid) to allow gradient descent optimization (PPO).
    
    Paper Ref: Eq. 4: mu_eta(x) = sigmoid(alpha * (beta * x - phi))
    """
    def __init__(self, input_dim, output_dim, depth=4):
        super(DifferentiableDecisionTree, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_internal_nodes = 2 ** depth - 1
        
        # We model the splits as Linear layers followed by Sigmoid
        # Weights (beta) and Bias (-phi)
        self.split_layers = nn.ModuleList()
        
        # Construct layers for each depth level
        for d in range(depth):
            nodes_at_depth = 2 ** d
            layer = nn.Linear(input_dim, nodes_at_depth)
            self.split_layers.append(layer)
            
        # Leaf parameters (Action Logits or Value)
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, output_dim))
        
        # Inverse temperature parameter (alpha in the paper)
        self.alpha = nn.Parameter(torch.ones(1)) 

    def forward(self, x):
        batch_size = x.size(0)
        
        # Start with probability 1.0 at the root
        path_probs = torch.ones(batch_size, 1, device=x.device)
        
        for d in range(self.depth):
            splits = self.split_layers[d](x)
            decisions = torch.sigmoid(self.alpha * splits)
            
            prob_left = path_probs * decisions
            prob_right = path_probs * (1 - decisions)
            
            path_probs = torch.stack([prob_left, prob_right], dim=2).view(batch_size, -1)
            
        output = torch.matmul(path_probs, self.leaf_weights)
        return output

class DDTFeaturesExtractor(BaseFeaturesExtractor):
    """
    Identity feature extractor.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int):
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)

class DDTMlpExtractor(nn.Module):
    """
    A dummy extractor that satisfies SB3's interface requirements.
    It passes features through unchanged but provides the expected attributes.
    """
    def __init__(self, features_dim):
        super().__init__()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim
        
    def forward(self, features):
        # SB3 expects a tuple (latent_pi, latent_vf)
        return features, features

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return features

class DDTMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Custom Policy for Maskable PPO that uses Differentiable Decision Trees.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        ddt_depth=4,
        **kwargs
    ):
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
        """
        Build the DDTs and a dummy MLP extractor.
        """
        self.ortho_init = False
        input_dim = np.prod(self.observation_space.shape)
        
        # 1. Action Tree (Actor)
        if isinstance(self.action_space, spaces.Discrete):
            action_dim = self.action_space.n
        else:
            raise NotImplementedError("DDT example implementation assumes Discrete Action Space")

        self.ddt_actor = DifferentiableDecisionTree(
            input_dim=input_dim, 
            output_dim=action_dim, 
            depth=self.ddt_depth
        )
        
        # 2. Value Tree (Critic)
        self.ddt_critic = DifferentiableDecisionTree(
            input_dim=input_dim, 
            output_dim=1, 
            depth=self.ddt_depth
        )

        # 3. Dummy Extractor for SB3 compatibility
        # This fixes the AttributeError: 'Identity' object has no attribute 'latent_dim_pi'
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
        """
        Override _predict to use DDT actor explicitly.
        Standard SB3 _predict uses self.action_net (Linear) which we want to ignore.
        """
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
        """
        Robust method to get distribution. Handles cases where proba_distribution
        might not accept action_masks by applying the mask manually to logits.
        """
        try:
            # Try passing action_masks as kwarg (Standard sb3-contrib behavior)
            return self.action_dist.proba_distribution(action_logits, action_masks=action_masks)
        except TypeError:
            # Fallback: Distribution doesn't accept action_masks argument.
            # We must apply the mask manually to the logits.
            if action_masks is not None:
                # Ensure mask is a boolean tensor
                if isinstance(action_masks, list):
                    # Handle Python List (e.g., from manual evaluation loop)
                    action_masks = torch.tensor(action_masks, device=action_logits.device, dtype=torch.bool)
                elif isinstance(action_masks, np.ndarray):
                    # Handle NumPy Array
                    action_masks = torch.as_tensor(action_masks, device=action_logits.device, dtype=torch.bool)
                else:
                    # It is likely a torch tensor, ensure it is boolean
                    action_masks = action_masks.bool()

                
                # Apply mask: set invalid actions to a very large negative number
                # effectively making their probability 0 (e^-inf)
                HUGE_NEG = -1e8
                action_logits = torch.where(action_masks, action_logits, torch.tensor(HUGE_NEG, device=action_logits.device))
            
            # Create distribution with masked logits
            distribution = self.action_dist.proba_distribution(action_logits)
            
            # Try to attach the mask to the distribution object if possible
            # (Used for entropy calculation in some implementations)
            if action_masks is not None and hasattr(distribution, "action_masks"):
                distribution.action_masks = action_masks
                
            return distribution

def export_discrete_tree(ddt_model, feature_names=None, action_names=None):
    """
    Converts the soft-differentiable tree into a hard interpretable tree 
    by discretizing weights as described in the paper.
    """
    print("\n=== Extracted Interpretable Decision Tree ===")
    
    # Recursively print the tree
    # We follow the path logic: index 0 is root.
    # Left child of index i is 2*i, Right child is 2*i + 1 (in BFS/heap order)
    # But our layer structure is: split_layers[d] has 2^d nodes.
    
    def print_node(depth, node_idx_in_layer, prefix=""):
        if depth == ddt_model.depth:
            # LEAF NODE
            # Calculate global leaf index
            # This logic assumes the standard tree expansion order
            global_leaf_idx = node_idx_in_layer
            weights = ddt_model.leaf_weights[global_leaf_idx].detach().cpu().numpy()
            best_action = np.argmax(weights)
            action_name = action_names[best_action] if action_names else str(best_action)
            print(f"{prefix}└── Action: {action_name} (Logits: {weights})")
            return

        # INTERNAL NODE
        layer = ddt_model.split_layers[depth]
        # Extract weights for this specific node in the vectorized layer
        # Layer weight shape: [nodes_at_depth, input_dim]
        w = layer.weight[node_idx_in_layer].detach().cpu().numpy()
        b = layer.bias[node_idx_in_layer].detach().cpu().item()
        
        # Discretization Strategy (Section 4.1 in paper):
        # 1. Argmax over beta (weights) to find the feature used for split
        feature_idx = np.argmax(np.abs(w))
        feature_val = w[feature_idx]
        
        # 2. Normalize threshold: phi / beta
        # The equation is sigmoid(beta * x + bias). 
        # Decision boundary is beta * x + bias = 0 => x = -bias / beta
        threshold = -b / feature_val
        
        fname = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
        
        # Branch logic
        print(f"{prefix}├── IF {fname} > {threshold:.4f}:")
        print_node(depth + 1, node_idx_in_layer * 2, prefix + "│   ")
        
        print(f"{prefix}├── ELSE:")
        print_node(depth + 1, node_idx_in_layer * 2 + 1, prefix + "│   ")

    # Start recursion from root (Depth 0, Index 0)
    print_node(0, 0)
    print("=============================================\n")

# ==========================================
# Example Usage
# ==========================================

def mask_fn(env):
    """
    Dummy masking function.
    """
    return [True, True] 

def make_env():
    """Create a wrapped environment that supports masking"""
    env = gym.make("CartPole-v1")
    env = ActionMasker(env, mask_fn)
    return env

if __name__ == "__main__":
    env = make_env()

    # CartPole Feature Names for interpretation
    feature_names = ["Cart Pos", "Cart Vel", "Pole Angle", "Pole Vel"]
    action_names = ["Left", "Right"]

    model = MaskablePPO(
        DDTMaskableActorCriticPolicy,
        env,
        verbose=1,
        learning_rate=1e-3,
        policy_kwargs=dict(ddt_depth=3)
    )

    print("Training DDT Policy (Increased timesteps for convergence)...")
    # Increased from 5000 to 20000 to allow the tree to converge better
    model.learn(total_timesteps=20000)

    print("\nEvaluating...")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action_masks = mask_fn(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        done = done or truncated

    print(f"Total Reward: {total_reward}")

    # Interpret the trained model
    export_discrete_tree(model.policy.ddt_actor, feature_names, action_names)