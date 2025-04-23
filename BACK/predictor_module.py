import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            average_precision_score, classification_report)
from torch.utils.data import DataLoader, TensorDataset
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

CONFIG = {
    'random_seed': 41,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 200,
    'patience': 10,
    'hidden_layers': [64, 48, 32, 24, 16],
    'dropout_rate': 0.3,
    'weight_decay': 1e-4,
    'n_splits': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Set seeds
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class FaultPredictionModel(nn.Module):
    def __init__(self, in_features, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_size = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(prev_size, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden(x)
        return self.out(x)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna().drop_duplicates()
    df['bug'] = df['bug'].astype(int)
    X = df.drop('bug', axis=1).values
    y = df['bug'].values
    return X, y, df.columns[:-1]

def create_dataloaders(X_train, X_val, y_train, y_val, batch_size):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    return train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            val_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu())
            all_preds.extend(preds.cpu())
            all_targets.extend(batch_y.cpu())
    return val_loss / len(loader), torch.stack(all_preds), torch.stack(all_targets), torch.stack(all_probs)

def calculate_metrics(y_true, y_pred, y_probs):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    y_probs_np = y_probs.numpy()
    return {
        'accuracy': accuracy_score(y_true_np, y_pred_np),
        'roc_auc': roc_auc_score(y_true_np, y_probs_np),
        'average_precision': average_precision_score(y_true_np, y_probs_np),
        'recall': recall_score(y_true_np, y_pred_np, zero_division=0),
        'precision': precision_score(y_true_np, y_pred_np, zero_division=0),
        'f1': f1_score(y_true_np, y_pred_np, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true_np, y_pred_np),
        'classification_report': classification_report(y_true_np, y_pred_np, zero_division=0, output_dict=True)
    }

def run_prediction(filepath):
    X, y, feature_names = load_and_preprocess_data(filepath)
    criterion = FocalLoss(alpha=1.5)
    kfold = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['random_seed'])

    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        train_loader, val_loader = create_dataloaders(X_train, X_val, y_train, y_val, CONFIG['batch_size'])

        model = FaultPredictionModel(
            in_features=X_train.shape[1],
            hidden_layers=CONFIG['hidden_layers'],
            dropout_rate=CONFIG['dropout_rate']
        ).to(CONFIG['device'])

        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

        best_ap = 0
        no_improve = 0

        for epoch in range(CONFIG['epochs']):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
            val_loss, val_preds, val_targets, val_probs = validate(model, val_loader, criterion, CONFIG['device'])
            avg_precision = average_precision_score(val_targets.numpy(), val_probs.numpy())

            if avg_precision > best_ap:
                best_ap = avg_precision
                no_improve = 0
                torch.save(model.state_dict(), f'best_model_fold{fold + 1}.pth')
            else:
                no_improve += 1
                if no_improve >= CONFIG['patience']:
                    break

        model.load_state_dict(torch.load(f'best_model_fold{fold + 1}.pth'))
        val_loss, val_preds, val_targets, val_probs = validate(model, val_loader, criterion, CONFIG['device'])
        metrics = calculate_metrics(val_targets, val_preds, val_probs)
        all_metrics.append(metrics)

    return all_metrics
