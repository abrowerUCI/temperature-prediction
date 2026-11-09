import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

class TempNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, dropout_rate=0.2):
        super(TempNN, self).__init__()
        
        # Structural layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Process the forward pass step-by-step
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TempLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=1, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def predict(model, X, y_scaler):
    """
    Uses the model to predict y in Degrees C.

    Args:
        model (PyTorch Model): PyTorch model used for prediction
        X (2-d dataframe): 2d dataframe containing the predictor values
        y_scaler (sklearn scaler): scaler object used for y values
    
        Return:
            y_pred (predicted temperature values)
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.as_tensor(X, dtype=torch.float32)).numpy()
    
    return y_scaler.inverse_transform(y_pred).ravel()

def eval(model, X, y, y_scaler, baseline):
    """
    Evaluates a PyTorch model.

    Args:
        model (PyTorch Model): PyTorch model being evaluated
        X (2-d dataframe): 2d dataframe containing the predictor values
        y (1-d series): list containing true y values
        y_scaler (sklearn scaler): scaler object used for y values
        baseline (float): baseline RMSE
    
        Return:
            r2, rmse, skill
    """
    y = y_scaler.inverse_transform(y).ravel()

    y_pred = predict(model, X, y_scaler)

    r2 = r2_score(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    skill = 1 - (rmse / baseline)

    return r2, rmse, skill