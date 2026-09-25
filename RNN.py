import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import copy
from itertools import product

from sklearn.model_selection import TimeSeriesSplit

# implementation of the three RNNs
class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, activation=torch.tanh):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = nn.Linear(input_size+hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, input):
        batch, seq_len, _ = input.shape
        h = input.new_zeros(batch, self.hidden_size)
        for t in range(seq_len):
            context_unit = h
            h = self.activation(self.hidden(torch.cat((context_unit, input[:, t]), dim=1)))
        return self.output(h)

class JordanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, activation=torch.tanh):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = nn.Linear(input_size+output_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, input):
        batch, seq_len, _ = input.shape
        y = input.new_zeros(batch, self.output_size)
        for t in range(seq_len):
            state_unit = y
            h = self.activation(self.hidden(torch.cat((state_unit, input[:, t]), dim=1)))
            y = self.output(h)
        return y

class MultiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, activation=torch.tanh):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = nn.Linear(input_size+hidden_size+output_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, input):
        batch, seq_len, _ = input.shape
        h = input.new_zeros(batch, self.hidden_size)
        y = input.new_zeros(batch, self.output_size)
        for t in range(seq_len):
            context_unit = h
            h = self.activation(self.hidden(torch.cat((context_unit, y, input[:, t]), dim=1)))
            y = self.output(h)
        return y

# given a time series: generate training windows
def make_windows(s, p):
    X = np.array([s[i:i + p] for i in range(len(s) - p)])
    y = s[p:].reshape(-1, 1)
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

# train a given model on a given window
def train(model, X, y, X_val, y_val, epochs=1000, patience=50, lr=1e-2, wd=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) # used adam optimiser
    loss_fn = nn.MSELoss() # used mse loss
    best, best_state, wait = np.inf, None, 0 # early stopping logic
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
        with torch.no_grad():
            val = loss_fn(model(X_val), y_val).item()
        if val < best:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best

# performance metrics
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

# model running functions
from sklearn.preprocessing import MinMaxScaler


def run_fold(series, tr, te, Model, p=6, n_hidden=8, lr=1e-2, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    # rescale the values for tanh activation
    scaler = MinMaxScaler((-1, 1)).fit(series[tr].reshape(-1, 1))
    s = scaler.transform(series.reshape(-1, 1)).ravel()
    # generate windows
    X, y = make_windows(s, p)
    k, m = te[0], te[-1] + 1
    X_tr, y_tr = X[:k - p], y[:k - p]
    X_te = X[k - p:m - p]
    # early stopping (final fifth of windw)
    n_val = max(1, len(X_tr) // 5)
    model, val_loss = train(Model(input_size=1, hidden_size=n_hidden), X_tr[:-n_val], y_tr[:-n_val], X_tr[-n_val:], y_tr[-n_val:],lr=lr, wd=wd)
    with torch.no_grad():
        pred = scaler.inverse_transform(model(X_te).numpy()).ravel()
    actual = series[k:m]
    naive = series[k - 1:m - 1] # the naive model predicts the current value as the next value
    return {"val_loss": val_loss, "rmse": rmse(actual, pred), "naive_rmse": rmse(actual, naive), "pred": pred, "idx": np.arange(k, m)}

def tune(series, Model, grid, n_splits=3):
    rows = []
    for p, h, lr, wd in product(*grid.values()):
        val = np.mean([run_fold(series, tr, te, Model, p=p, n_hidden=h, lr=lr, wd=wd)["val_loss"] for tr, te in TimeSeriesSplit(n_splits=n_splits).split(series)])
        rows.append({"p": p, "n_hidden": h, "lr": lr, "wd": wd, "val_loss": val})
    return pd.DataFrame(rows).sort_values("val_loss")