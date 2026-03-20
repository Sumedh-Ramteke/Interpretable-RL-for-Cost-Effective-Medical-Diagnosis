import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# import data_loader as dl
# import Imputation as imp

class Classifier():
        def __init__(self, dim, num_classes, clf_para):
                '''
                clf_para: {'hidden_size':, 'lr':, 'batch_size', 'save_dir'}
                '''
                self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
                print(self.device)

                self.para = clf_para
                self.save_dir = clf_para['save_dir']

                self.feature_size = dim
                self.num_classes =  num_classes

                #unique, counts = np.unique(self.train[:,-1], return_counts=True) # TODO: what is the order of counts? counts[0] is the number of zero?
                # self.class_weights = th.tensor([1 / (self.para['class_weights'] + 1), 1 - 1 / (self.para['class_weights'] + 1)]).float()

                #self.class_weights = 1 - th.tensor(counts / len(self.train)).float()
                # print(1 - th.tensor(np.unique(self.train[:,-1], return_counts=True)[1] / len(self.train)).float())
                # print(1 - th.tensor(np.unique(self.val[:,-1], return_counts=True)[1] / len(self.val)).float())
                # print(1 - th.tensor(np.unique(self.test[:,-1], return_counts=True)[1] / len(self.test)).float())

                #self.class_weights = 1 - th.tensor(counts / len(self.train)).float()
                self.class_weights = th.tensor(self.para['class_weights']).float()  # I set class_weights to a hyper parameter here. TODO
                self.class_weights = self.class_weights.to(self.device)

                self.hidden_size = self.para['hidden_size']
                self.lr = self.para['lr']
                self.batch_size = self.para['batch_size']


                # initialize the model
                self.model = DifferentiableDecisionTree(self.feature_size, self.num_classes).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight = self.class_weights) # is it better to 1/weigtht? .i.e., for minor class, the weight should be large.

                return

        def set_dataset(self, train, val, test):
                self.train = train.astype('float')
                self.val = val.astype('float')
                self.test = test.astype('float')
                self.train_dl = DataLoader(dataset = clf_data(self.train), batch_size = self.batch_size, shuffle=True, drop_last=True)
                self.val_dl = DataLoader(dataset = clf_data(self.val), batch_size = self.batch_size)
                self.test_dl = DataLoader(dataset = clf_data(self.test), batch_size = self.batch_size)

                assert self.feature_size == len(train[0]) - 1, 'wrong feature size!'
                unique, counts = np.unique(self.train[:,-1], return_counts=True) # TODO: what is the order of counts? counts[0] is the number of zero?
                assert self.num_classes == len(unique), 'wrong num_classes!'


        def train_model(self, dl, epochs, verbose = 0, fresh = True):
                ## train for one epoch step dl
                size = len(dl.dataset)
                self.optimizer = th.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay=1e-2)
                
                if fresh:  # fresh training
                    if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                    writer = SummaryWriter(self.save_dir + '/' + 'tensorboard')


                for t in range(epochs):
                        if t>0:
                                if verbose != 0:
                                        print(f"Epoch {t+1}\n-------------------------------")
                                self.model.train()
                                for batch, (X, y) in enumerate(dl):
                                        X, y = X.to(self.device), y.to(self.device)

                                # Compute prediction error
                                        pred = self.model(X)
                                        loss = self.criterion(pred, y)

                                        # Backpropagation
                                        self.optimizer.zero_grad()
                                        loss.backward()
                                        self.optimizer.step()

                                        if verbose != 0 and batch % 100 == 0:
                                                loss, current = loss.item(), batch * len(X)
                                                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                        if fresh:
                            train_loss, train_balanced_accuracy, train_auroc, train_f1, train_recall, train_precision = self.test_model(self.train_dl, verbose = verbose)
                            val_loss, val_balanced_accuracy, val_auroc, val_f1, val_recall, val_precision = self.test_model(self.val_dl, verbose = verbose)
                            test_loss, test_balanced_accuracy, test_auroc, test_f1, test_recall, test_precision = self.test_model(self.test_dl, verbose = verbose)

                            writer.add_scalars('loss', {'train': train_loss, 'val': val_loss, 'test': test_loss}, t)
                            writer.add_scalars('balanced_accuracy', {'train': train_balanced_accuracy, 'val': val_balanced_accuracy, 'test': test_balanced_accuracy}, t)
                            writer.add_scalars('auroc', {'train': train_auroc, 'val': val_auroc, 'test': test_auroc}, t)
                            writer.add_scalars('f1', {'train': train_f1, 'val': val_f1, 'test': test_f1}, t)
                            writer.add_scalars('recall', {'train': train_recall, 'val': val_recall, 'test': test_recall}, t)
                            writer.add_scalars('precision', {'train': train_precision, 'val': val_precision, 'test': test_precision}, t)

                if fresh:
                    # final test & save model & report in tensorboard
                    th.save(self.model.state_dict(), self.save_dir + '/cls_model.pth')
                    metric={'loss':val_loss, 'bacc': val_balanced_accuracy, 'auroc': val_auroc, 'f1': val_f1, 'recall': val_recall, 'prec': val_precision,
                            'test_loss':test_loss, 'test_bacc': test_balanced_accuracy, 'test_auroc': test_auroc, 'test_f1': test_f1, 'test_recall': test_recall, 'test_prec': test_precision}
                    del self.para['save_dir']
                    self.para['class_weight_ratio'] = self.para['class_weights'][0] / self.para['class_weights'][1]
                    del self.para['class_weights']
                    writer.add_hparams(self.para,metric,name='hparam')
                    writer.close()

                return

        def test_model(self, dl, verbose = 0):
                ## test self.model on dl
                size = len(dl.dataset)
                num_batches = len(dl)

                self.model.eval()
                test_loss, accuracy = 0, 0

                pred_prob = th.empty(0).float().to(self.device)
                pred_label = th.empty(0).float().to(self.device) # TODO: should change to Long
                true_label = th.empty(0).float().to(self.device) # TODO: should change to Long
                # true_label = th.empty(1)

                with th.no_grad():
                        for X, y in dl:
                                X, y = X.to(self.device), y.to(self.device)
                                pred = self.model(X)

                                pred_prob = th.cat((pred_prob, pred[:, -1]), 0)
                                pred_label = th.cat((pred_label, pred.argmax(1)), 0)
                                true_label = th.cat((true_label, y), 0)

                                test_loss += self.criterion(pred, y).item()
                                accuracy += (pred.argmax(1) == y).type(th.float).sum().item()


                test_loss /= num_batches
                accuracy /= size

                pred_label = pred_label.cpu().numpy()
                true_label = true_label.cpu().numpy()
                pred_prob = pred_prob.cpu().numpy()
                auroc = roc_auc_score(true_label, pred_prob)
                precision = precision_score(true_label, pred_label)
                recall = recall_score(true_label, pred_label)
                f1 = f1_score(true_label, pred_label)
                balanced_accuracy = balanced_accuracy_score(true_label, pred_label)

                if verbose != 0:
                        print(f"Test Error: \n Balanced Accuracy: {(100*balanced_accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, AUROC: {auroc:>0.3f}, F1: {f1:>0.3f}, Recall: {recall:>0.3f}, Precision: {precision:>0.3f} \n")

                return test_loss, balanced_accuracy, auroc, f1, recall, precision

        @th.no_grad()
        def predict(self, data):
                ## using self.model to transform data
                # return self.model(data)
                data = data.to(self.device)
                return nn.functional.softmax(self.model(data), dim = -1)

        # def predict(self, data):
        #       ## using self.model to transform data
        #       data = th.tensor(data).float()
        #       return self.model(data).detach().numpy()


        def model_save(self, filename, save_dir = None):
                ## save imputation model
                if save_dir is None:
                    save_dir = './clf_model'
                th.save(self.model.state_dict(), save_dir + '/' + filename + '.pth')
                return

        def model_load(self, filename, load_dir = None):
                ## load imputation model
                if load_dir is None:
                    load_dir = './clf_model'
                self.model.load_state_dict(th.load(load_dir + '/' + filename + '.pth', map_location=self.device))
                return

# softmax DNN predictor
class DifferentiableDecisionTree(nn.Module):
    def __init__(self, input_dim, output_dim=2, depth=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.num_leaves = 2 ** depth

        # Decision layers: each level has nodes that each produce a soft decision
        self.decision_layers = nn.ModuleList()
        curr_nodes = 1
        for d in range(depth):
            self.decision_layers.append(nn.Linear(input_dim, curr_nodes))
            curr_nodes *= 2

        # Leaf logits
        self.leaf_weights = nn.Parameter(th.randn(self.num_leaves, output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        path_probs = th.ones(batch_size, 1, device=x.device)

        for level in range(self.depth):
            decisions = th.sigmoid(self.decision_layers[level](x))
            p_left = 1 - decisions
            p_right = decisions

            combined = th.stack([p_left, p_right], dim=2).view(batch_size, -1)
            path_probs = path_probs.repeat_interleave(2, dim=1) * combined

        out = path_probs @ self.leaf_weights
        return out


class clf_data(Dataset):
        def __init__(self, data):
                self.x=th.tensor(data[:,:-1], dtype=th.float)
                self.y=th.tensor(data[:,-1], dtype=th.long)
                self.len=self.x.shape[0]
        def __getitem__(self,index):
                return self.x[index], self.y[index]
        def __len__(self):
                return self.len