import numpy as np
import pandas as pd
import torch
import imputer.flow_models as flow_models
import imputer.nflow as nf
from sklearn import model_selection
from sklearn import preprocessing

import time
from tqdm import tqdm
import matplotlib.pyplot as plt


class Imputer():  # mean_imputer
    def __init__(self, dim, impute_para):
        self.d = dim
        self.batch_size = impute_para['batch_size']
        self.lr = impute_para['lr']
        self.alpha = impute_para['alpha']
        # [CUDA] Define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Imputer using device: {self.device}")

    def set_dataset(self, train, mask):
        '''
        train: complete data with no label information
        mask: 0-1 indicator of same shape, 1 means missing, 0 means observed
        '''
        # Keep large dataset on CPU to save VRAM, move batches to GPU later
        self.X = torch.tensor(train).float()
        self.mask = torch.tensor(mask).int()

        # Generate mask and X_hat
        assert self.d == self.X.shape[1], 'wrong dim'
        self.X_hat = ((1 - self.mask) * self.X).float()

        # [CUDA] Compute stats and move to GPU immediately
        mu_hat = torch.mean(self.X, dim=0).float().to(self.device)
        Sigma_hat = torch.cov(self.X.T).float().to(self.device)

        # Ensure Positive Definite at initialization
        L, V = torch.linalg.eig(Sigma_hat) 
        if torch.min(L.real) < 1e-4:
            print('Init: Not PD, adding jitter...')
            eye_matrix = torch.eye(self.d).to(self.device)
            jitter = (1e-4 - torch.min(L.real))
            Sigma_hat = torch.add(Sigma_hat, eye_matrix * jitter)

        self.init_model()

        # [CUDA] Create distribution on GPU
        q0 = torch.distributions.multivariate_normal.MultivariateNormal(mu_hat, covariance_matrix=Sigma_hat)
        self.nfm.q0 = q0

        return

    def init_model(self):
        # Construct flow model
        num_flows = 32
        torch.manual_seed(0)

        flows = []
        for i in range(num_flows):
            if self.d % 2 == 1:
                param_map = flow_models.MLP([self.d // 2 + 1, 32, 32, self.d], init_zeros=True)
            else:
                param_map = flow_models.MLP([self.d // 2, 32, 32, self.d], init_zeros=True)
            flows.append(nf.AffineCouplingBlock(param_map))
            flows.append(nf.Permute(self.d, mode='swap'))
        
        self.nfm = nf.NormalizingFlow(q0=None, flows=flows).to(self.device)

    def generate_hat_mask(self, data, mask):
        if data is None:
            return self.X, self.X_hat, self.mask

        if data.ndim == 1:
            data = data.reshape((1, len(data)))
            mask = mask.reshape((1, len(data)))

        X = torch.tensor(data).float()
        mask = torch.tensor(mask).int()
        X_hat = ((1 - mask) * X).float()

        return X, X_hat, mask

    def train_model(self, data=None, mask=None, max_iter=10):
        batch_size = self.batch_size
        lr = self.lr
        alpha = self.alpha
        rho = 0.99 * np.arange(1, max_iter + 1) ** (-0.8)
        # beta = 1e-3 / np.arange(1, max_iter + 1)

        d = self.d
        nfm = self.nfm
        mu_hat = nfm.q0.loc
        Sigma_hat = nfm.q0.covariance_matrix

        X_true_train, X_hat_train, mask_train = self.generate_hat_mask(data, mask)
        
        # Safety check: If we have fewer samples than batch_size, reduce batch_size
        current_batch_size = min(batch_size, X_hat_train.shape[0])
        if current_batch_size < 2: 
            print("Warning: Not enough data to train imputer (Need >1). Skipping.")
            return

        optimizer = torch.optim.Adam(self.nfm.parameters(), lr=lr, weight_decay=1e-6)

        for J in tqdm(range(max_iter)):
            nfm.zero_grad()

            batch_ind = np.random.choice(X_hat_train.shape[0], size=current_batch_size, replace=(X_hat_train.shape[0] < current_batch_size))
            x_hat = X_hat_train[batch_ind, :]
            x_true = X_true_train[batch_ind, :]
            mask_batch = mask_train[batch_ind, :]

            # [CUDA] Move batches to GPU
            x_hat = x_hat.to(self.device)
            x_true = x_true.to(self.device)
            mask_batch = mask_batch.to(self.device).float()

            # 1. Update flow model
            log_prob = nfm.log_prob(x_hat)
            L1 = -torch.mean(log_prob)
            
            # [Fix] Check for NaNs in loss
            if torch.isnan(L1):
                print("Warning: L1 loss is NaN. Skipping step.")
                optimizer.zero_grad()
                continue
                
            L1.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 2. Update base distribution
            z = nfm.inverse(x_hat)

            Sigma_mask_mo = mask_batch.view(current_batch_size, d, 1) @ (1 - mask_batch).view(current_batch_size, 1, d)
            Sigma_mask_oo = (1 - mask_batch).view(current_batch_size, d, 1) @ (1 - mask_batch).view(current_batch_size, 1, d)
            
            # Group by unique mask patterns to avoid redundant matrix inversions
            # With block-level masking there are at most 2^n_panels << batch_size unique patterns
            unique_masks, inverse_idx = torch.unique(mask_batch, dim=0, return_inverse=True)
            inv_Sigma_oo = torch.zeros(current_batch_size, d, d, device=self.device)
            for ui in range(len(unique_masks)):
                mask_u = unique_masks[ui]
                Sigma_mask_oo_u = (1 - mask_u).view(d, 1) @ (1 - mask_u).view(1, d)
                inv_u = self.inverse_masked(Sigma_hat, Sigma_mask_oo_u)
                sample_indices = (inverse_idx == ui).nonzero(as_tuple=True)[0]
                inv_Sigma_oo[sample_indices] = inv_u
            
            z_m = mu_hat * mask_batch + ((Sigma_hat.unsqueeze(0) * Sigma_mask_mo) @ inv_Sigma_oo @ (
                        (z - mu_hat) * (1 - mask_batch)).view(current_batch_size, d, 1)).squeeze()
            z_hat = z * (1 - mask_batch) + z_m
            
            # Clamp z_hat to prevent explosion
            z_hat = torch.clamp(z_hat, -10.0, 10.0)

            Sigma_m = Sigma_hat * (mask_batch.view(current_batch_size, d, 1) @ mask_batch.view(current_batch_size, 1, d)) \
                      - (Sigma_hat * Sigma_mask_mo) @ inv_Sigma_oo @ (
                                  Sigma_hat * ((1 - mask_batch).view(current_batch_size, d, 1) @ mask_batch.view(current_batch_size, 1, d)))

            # Compute local -> global mu and Sigma
            mu_hat_local = torch.mean(z_hat, dim=0)
            
            # [Fix] Detach z_hat for covariance calculation to save memory/stability
            z_diff = (z_hat - mu_hat).detach()
            Sigma_hat_local = torch.mean(z_diff.view(current_batch_size, d, 1) @ z_diff.view(current_batch_size, 1, d) + Sigma_m, dim=0)
            
            mu_hat = rho[J] * mu_hat_local + (1 - rho[J]) * mu_hat
            Sigma_hat = rho[J] * Sigma_hat_local + (1 - rho[J]) * Sigma_hat

            # Initialize new base distribution
            mu_hat = mu_hat.detach()
            Sigma_hat = Sigma_hat.detach()
            Sigma_hat = (Sigma_hat + torch.t(Sigma_hat)) / 2

            # [Fix] Robust PD Check inside loop
            L, V = torch.linalg.eig(Sigma_hat)
            min_eig = torch.min(L.real)
            # Increased threshold to 1e-4 for better stability
            if min_eig < 1e-4:
                jitter = 1e-4 - min_eig
                Sigma_hat = Sigma_hat + torch.eye(d).to(self.device) * jitter

            new_base = torch.distributions.multivariate_normal.MultivariateNormal(mu_hat, covariance_matrix=Sigma_hat)
            nfm.init_base(new_base)

            # 3. Update flow model again
            x_tilde = nfm(z_hat)
            log_prob_tilde = nfm.log_prob(x_tilde)
            L_rec = torch.sum(torch.pow(x_tilde - x_true, 2), dim=1)
            L2 = -torch.mean(log_prob_tilde - alpha * L_rec)
            
            if torch.isnan(L2):
                print("Warning: L2 loss is NaN. Skipping step.")
                continue

            L2.backward()
            optimizer.step()

        return

    def inverse_masked(self, A, mask):
        """
        Robust inverse of sub-matrix corresponding to observed values.
        """
        sub_matrix = torch.masked_select(A, mask.bool())
        dim_sub = int(np.sqrt(sub_matrix.numel())) # Safer dimension calc
        sub_matrix = sub_matrix.view(dim_sub, dim_sub)
        
        # [CRITICAL FIX] Add Jitter to diagonal to prevent singular matrix inversion
        # Correlated features (like BP and MAP) make this matrix singular without jitter.
        jitter = 1e-4 * torch.eye(dim_sub, device=self.device)
        sub_matrix = sub_matrix + jitter
        
        try:
            inv_mat = torch.inverse(sub_matrix)
        except RuntimeError:
            # Fallback if inverse still fails
            inv_mat = torch.linalg.pinv(sub_matrix)

        B = torch.zeros(A.shape, device=self.device)
        B[mask.bool()] = inv_mat.flatten().float()
        return B

    def transform(self, data=None):
        with torch.no_grad():
            d = self.d
            nfm = self.nfm
            if data is None:
                X_hat, mask = self.X_hat, self.mask
            else:
                if isinstance(data, torch.Tensor):
                    X_hat = data.float()
                    if X_hat.dim() == 1:
                        X_hat = X_hat.unsqueeze(0)
                else:
                    if data.ndim == 1:
                        data = data.reshape((1, len(data)))
                    X_hat = torch.tensor(data).float()
                mask = torch.isnan(X_hat).int()
                X_hat = torch.nan_to_num(X_hat).float()

            X_hat = X_hat.to(self.device)
            mask = mask.to(self.device).float()
            n = X_hat.shape[0]

            Z = nfm.inverse(X_hat)

            mu_hat = nfm.q0.loc
            Sigma_hat = nfm.q0.covariance_matrix

            Sigma_mask_mo = mask.view(n, d, 1) @ (1 - mask).view(n, 1, d)
            Sigma_mask_oo = (1 - mask).view(n, d, 1) @ (1 - mask).view(n, 1, d)
            
            # Group by unique mask patterns to avoid redundant matrix inversions
            unique_masks, inverse_idx = torch.unique(mask, dim=0, return_inverse=True)
            inv_Sigma_oo = torch.zeros(n, d, d, device=self.device)
            for ui in range(len(unique_masks)):
                mask_u = unique_masks[ui]
                Sigma_mask_oo_u = (1 - mask_u).view(d, 1) @ (1 - mask_u).view(1, d)
                inv_u = self.inverse_masked(Sigma_hat, Sigma_mask_oo_u).float()
                sample_indices = (inverse_idx == ui).nonzero(as_tuple=True)[0]
                inv_Sigma_oo[sample_indices] = inv_u
            
            Z_m = mu_hat * mask + ((Sigma_hat.unsqueeze(0) * Sigma_mask_mo) @ inv_Sigma_oo @ (
                        (Z - mu_hat) * (1 - mask)).view(n, d, 1)).squeeze(-1)
            Z_hat = Z * (1 - mask) + Z_m

            X_tilde = nfm(Z_hat)
            X_hat = X_hat * (1 - mask) + X_tilde * mask

            return X_hat

    def transform_batch_same_mask(self, data):
        """
        Optimized transform for batches where ALL samples share the same mask pattern.
        Computes the matrix inverse only ONCE for the entire batch instead of per-sample.
        Used for pre-computing imputer caches.
        """
        with torch.no_grad():
            d = self.d
            nfm = self.nfm

            if isinstance(data, torch.Tensor):
                X_hat = data.float()
            else:
                if data.ndim == 1:
                    data = data.reshape((1, len(data)))
                X_hat = torch.tensor(data).float()

            mask = torch.isnan(X_hat).int()
            X_hat = torch.nan_to_num(X_hat).float()

            X_hat = X_hat.to(self.device)
            mask = mask.to(self.device).float()

            batch_size = X_hat.shape[0]

            Z = nfm.inverse(X_hat)

            mu_hat = nfm.q0.loc
            Sigma_hat = nfm.q0.covariance_matrix

            # All samples share the same mask — compute inverse only ONCE
            mask_single = mask[0:1]  # (1, d)

            Sigma_mask_mo = mask_single.view(1, d, 1) @ (1 - mask_single).view(1, 1, d)
            Sigma_mask_oo = (1 - mask_single).view(1, d, 1) @ (1 - mask_single).view(1, 1, d)

            inv_Sigma_oo = self.inverse_masked(Sigma_hat, Sigma_mask_oo[0]).unsqueeze(0)

            Z_m = mu_hat * mask + (
                (Sigma_hat.unsqueeze(0) * Sigma_mask_mo) @ inv_Sigma_oo @
                ((Z - mu_hat) * (1 - mask)).view(batch_size, d, 1)
            ).squeeze(-1)

            Z_hat = Z * (1 - mask) + Z_m

            X_tilde = nfm(Z_hat)
            X_hat = X_hat * (1 - mask) + X_tilde * mask

            return X_hat

    def model_save(self, file_name, save_dir=None):
        if save_dir is None:
            save_dir = './imp_model'
        self.nfm.save(save_dir + '/' + file_name + '_merged.pth')
        return

    def model_load(self, file_name, load_dir=None):
        self.init_model()
        if load_dir is None:
            load_dir = './imp_model'
        self.nfm.load(load_dir + '/' + file_name + '_merged.pth')
        return