import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def to_sequence(features, labels):
    # One day per sample, three features. Shape (samples, 1, 3).
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(1)
    y_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    return x, y_tensor

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val, 
    criterion=nn.MSELoss(),
    epochs=300,
    batch_size=32,
    lr=1e-3,
    patience=25,
    lr_decay=1.0,
):
    """
    Train a regression network and restore the weights with the best validation loss.
    """

    def as_batch(features, labels):
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (samples, 1, features) for the LSTM
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        return x, y

    X_tr, y_tr = as_batch(X_train, y_train)
    X_va, y_va = as_batch(X_val, y_val)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    stopped = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped = epoch
                break
        scheduler.step()

    model.load_state_dict(best_state)
    model.eval()
    return model, stopped

def make_windows(features, labels, history):
    """
    Stack the previous `history` days into each sample.

    Sample t uses days [t - history + 1, t] and the label already stored on day t.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    xs = [
        features[t - history + 1 : t + 1]
        for t in range(history - 1, len(features))
    ]
    ys = [labels[t] for t in range(history - 1, len(labels))]
    return np.stack(xs), np.asarray(ys)