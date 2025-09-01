import numpy as np
import random
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def generate_reservoir(dim_reservoir=300, sigma=0.1, rho=1.2, sparsity=0.05, dim_system=128):
    def generate_win(sigma):
        q = dim_reservoir // dim_system
        win = np.zeros((dim_reservoir, dim_system))
        for i in range(dim_system):
            np.random.seed(i + 21)
            ip = sigma * (-1 + 2 * np.random.rand(q))
            win[(i * q):(i + 1) * q, i] = ip
        return win
    def generate_A(dim_reservoir, rho, sparsity):
        A = sparse.rand(dim_reservoir, dim_reservoir, density=sparsity, random_state=21).todense()
        vals = np.linalg.eigvals(A)
        e = np.max(np.abs(vals))
        A = (A / e) * rho
        return A

    W_in = generate_win(sigma)
    A = generate_A(dim_reservoir, rho, sparsity)
    return W_in, A

def iteration_reservoir(data, W_in, A):
    train_len = len(data[0])
    data = np.transpose(data)
    dim_reservoir = len(A)
    R = np.zeros((train_len, dim_reservoir))
    for i in range(1, train_len):
        R[i] = np.tanh(A.dot(R[i-1]) + W_in.dot(data[i-1]))
    r0 = np.tanh(A.dot(R[-1]) + W_in.dot(data[-1]))
    Rt = np.transpose(R)
    R2 = Rt.copy()
    for j in range(2, np.shape(R2)[0] - 2):
        if np.mod(j, 2) == 0:
            R2[j, :] = (Rt[j, :] * Rt[j, :]).copy()
    return R2, r0

def Ridge_regression(train_data, R, beta=1e-5):
    reservoir_size = R.shape[0]
    core_matrix = R @ R.T + beta * np.eye(reservoir_size)
    core_pinv = np.linalg.pinv(core_matrix)
    W_out = train_data @ R.T @ core_pinv
    return W_out

def predict_reservoir(test_data, W_in, A, r_0, W_out):
    dim_system, predict_len = len(test_data), len(test_data[0])
    output = np.zeros((dim_system, predict_len))
    r = np.transpose(r_0)
    for i in range(predict_len):
        r_2 = r.copy()
        for j in range(2, np.shape(r_2)[0]-2):
            if np.mod(j, 2) == 0:
                r_2[j] = (r[j] * r[j]).copy()
        out = np.squeeze(np.asarray(np.dot(W_out, r_2)))
        temp = np.tanh(np.squeeze(np.dot(A, r)) + np.dot(W_in, out))
        r = np.squeeze(np.asarray(temp))
    return output

def reservoir_layer(u, r, A, W_in):
    dr = np.tanh(np.dot(A, r) + np.dot(W_in, u))
    return dr

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.00001, mode="min", start_from_epoch=6):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.start_from_epoch = start_from_epoch
        self.best_score = None
        self.early_stop = False
        self.epoch = 0

    def __call__(self, val_loss, model):
        if self.mode == "min":
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.epoch += 1
            if self.epoch >= self.start_from_epoch and self.epoch >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.epoch = 0

    def save_checkpoint(self, val_loss, model):
        # Save model checkpoint
        torch.save(model.state_dict(), 'data/checkpoint.pt')

    def load_checkpoint(self, model):
        # Load model checkpoint
        model.load_state_dict(torch.load('data/checkpoint.pt'))

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.25, patience=5, min_delta=0.00001, min_lr=0.000025):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.num_bad_epochs = 0
        self.last_loss = None

    def __call__(self, val_loss):
        if self.last_loss is None or val_loss > self.last_loss - self.min_delta:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                new_lr = max(self.optimizer.param_groups[0]['lr'] * self.factor, self.min_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                self.num_bad_epochs = 0
        else:
            self.num_bad_epochs = 0
        self.last_loss = val_loss

def data_prepare(train_data, R):

    X = R[:, 100:]
    Y = train_data[:, 100:]

    train_val_separation = int(len(X.T) * 0.8)
    train_x_unscaled, val_x_unscaled = X[:, :train_val_separation], X[:, train_val_separation:]
    train_y_unscaled, val_y_unscaled = Y[:, :train_val_separation], Y[:, train_val_separation:]

    train_x = train_x_unscaled.T
    val_x = val_x_unscaled.T
    train_y = train_y_unscaled.T
    val_y = val_y_unscaled.T

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return trainloader, valloader, len(train_x[0]), len(train_y[0])

def data_prepare_2(train_data, R, seq_len=5):
    X = R[:, 100:]
    Y = train_data[:, 100:]

    train_val_separation = int(len(X.T) * 0.8)
    train_x_unscaled, val_x_unscaled = X[:, :train_val_separation], X[:, train_val_separation:]
    train_y_unscaled, val_y_unscaled = Y[:, :train_val_separation], Y[:, train_val_separation:]

    def create_sequences(data_x, data_y, seq_len):
        sequences_x, sequences_y = [], []
        for i in range(len(data_x) - seq_len):
            seq_x = data_x[i:i+seq_len]
            seq_y = data_y[i+1:i+seq_len+1]
            sequences_x.append(seq_x)
            sequences_y.append(seq_y)
        return np.array(sequences_x), np.array(sequences_y)

    train_x, train_y = create_sequences(train_x_unscaled.T, train_y_unscaled.T, seq_len)
    val_x, val_y = create_sequences(val_x_unscaled.T, val_y_unscaled.T, seq_len)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    trainloader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    valloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    return trainloader, valloader, train_x.shape[-1], train_y.shape[-1]

def iteration_reservoir_2(R, U, W_in, A):

    A_input = torch.matmul(R, A.t())
    W_in_output = torch.matmul(U, W_in.t())
    R_current = torch.tanh(A_input + W_in_output)
    indices = torch.arange(R_current.shape[1], device=R.device)
    mask = (indices >= 2) & (indices < R.shape[1] - 2) & (indices % 2 == 0)
    R2 = R_current.clone()
    R2[:, mask] = R_current[:, mask] ** 2

    return R2

