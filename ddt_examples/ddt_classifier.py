import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score

# Import the user-provided scripts
try:
    from blood_panel_data_preprocessing import sepsis_data
    from data_loader import Data_Loader
except ImportError as e:
    raise ImportError(f"Could not import required modules. Ensure 'blood_panel_data_preprocessing.py' and 'data_loader.py' are in the same directory. Error: {e}")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DifferentiableDecisionTree(nn.Module):
    """
    A Differentiable Decision Tree (DDT) implementation.
    
    Unlike standard decision trees (CART, ID3) which use hard splits (if x > 5 then left else right),
    a DDT uses soft splits (sigmoid functions). This allows the model to be differentiable 
    and trainable via Stochastic Gradient Descent (SGD), just like a neural network.
    """
    def __init__(self, input_dim, output_dim, depth=4):
        super(DifferentiableDecisionTree, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_internal_nodes = 2 ** depth - 1
        
        # Decision nodes: 
        # We learn a linear projection for each internal node: w*x + b
        # If result > 0 (sigmoid > 0.5), tend towards right child, else left.
        self.decision_layers = nn.ModuleList()
        
        curr_nodes = 1
        for d in range(depth):
            # Each node in this level needs its own set of weights
            self.decision_layers.append(nn.Linear(input_dim, curr_nodes))
            curr_nodes *= 2
            
        # Leaf parameters:
        # Each leaf holds a distribution over the classes (logits).
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        
        # probability of reaching the current node. 
        # Initialize with 1.0 for the root.
        path_probs = torch.ones(batch_size, 1).to(x.device)
        
        for level in range(self.depth):
            # Compute decision for all nodes at this level
            decisions = self.decision_layers[level](x)
            
            # Apply sigmoid to get probability of going RIGHT
            p_right = torch.sigmoid(decisions)
            p_left = 1.0 - p_right
            
            # Expand path_probs to match p_left/p_right dimensions for broadcasting
            current_probs_expanded = path_probs.repeat_interleave(2, dim=1)
            
            # Interleave left and right decisions to match the expanded path
            decisions_stacked = torch.stack([p_left, p_right], dim=2)
            decisions_flat = decisions_stacked.view(batch_size, -1)
            
            # Update path probabilities: P(Child) = P(Parent) * P(Decision)
            path_probs = current_probs_expanded * decisions_flat
            
        # Result = path_probs @ leaf_weights
        out = torch.matmul(path_probs, self.leaf_weights)
        
        return out

def main():
    # Configuration
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    TREE_DEPTH = 4
    
    try:
        # 1. Load Data using provided scripts
        print("Loading and preprocessing data via 'blood_panel_data_preprocessing'...")
        # sepsis_data typically returns (data, block, costs)
        # data contains X and y concatenated. y is the last column.
        dataset_content = sepsis_data()
        
        # Unpack based on length to be safe (handling potential variations in the script return)
        if len(dataset_content) == 3:
            data, block, costs = dataset_content
        else:
            data, block = dataset_content[0], dataset_content[1]
            
        print(f"Full dataset shape: {data.shape}")

        # 2. Use Data_Loader to split
        # The Data_Loader splits data into 4 parts: Imputer, Classifier, RL, Evaluation
        print("Initializing Data_Loader and splitting data...")
        loader = Data_Loader(data, block)
        
        # We need the 'Classifier' split.
        # Based on standard usage of this loader pattern:
        # loader.train_cls -> Training set for classifier
        # loader.test_cls  -> Testing set for classifier
        if hasattr(loader, 'train_cls') and hasattr(loader, 'test_cls'):
            train_data = loader.train_cls
            test_data = loader.test_cls
            print(f"Using 'train_cls' split. Train size: {train_data.shape}, Test size: {test_data.shape}")
        elif hasattr(loader, 'train') and hasattr(loader, 'test'):
            # Fallback if the 4-way split wasn't run or named differently
            train_data = loader.train
            test_data = loader.test
            print(f"Using standard 'train' split. Train size: {train_data.shape}, Test size: {test_data.shape}")
        else:
            raise AttributeError("Data_Loader object missing expected 'train_cls' or 'train' attributes.")

        # 3. Separate Features (X) and Target (y)
        # y is the last column in this dataset structure
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        
        # Convert to Tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)
        
        # 4. Initialize Model
        input_dim = X_train.shape[1]
        model = DifferentiableDecisionTree(input_dim=input_dim, output_dim=1, depth=TREE_DEPTH).to(device)
        
        # Loss and Optimizer
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        print(f"Training Differentiable Decision Tree (Depth {TREE_DEPTH})...")
        
        # 5. Training Loop
        model.train()
        for epoch in range(EPOCHS):
            permutation = torch.randperm(X_train_t.size()[0])
            
            epoch_loss = 0
            for i in range(0, X_train_t.size()[0], BATCH_SIZE):
                indices = permutation[i:i+BATCH_SIZE]
                batch_x, batch_y = X_train_t[indices], y_train_t[indices]
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(X_train_t):.4f}")
                
        # 6. Evaluation
        model.eval()
        with torch.no_grad():
            y_logits = model(X_test_t)
            y_probs = torch.sigmoid(y_logits).cpu().numpy()
            y_preds = (y_probs > 0.5).astype(int)
            y_true = y_test_t.cpu().numpy()
            
            acc = accuracy_score(y_true, y_preds)
            f1 = f1_score(y_true, y_preds) # Calculate F1 Score
            try:
                auc = roc_auc_score(y_true, y_probs)
            except ValueError:
                auc = 0.0 # Handle case where only one class is present in test batch
            
            print("\n" + "="*30)
            print("RESULTS")
            print("="*30)
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Score: {f1:.4f}") # Print F1 Score
            print(f"ROC AUC:  {auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_preds))
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()