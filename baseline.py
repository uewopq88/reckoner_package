"""
Baseline experiment (NO Reckoner framework).

Trains a single plain model with ordinary supervised learning -- no
identification / cold-start / refinement / noise -- and evaluates it with the
exact same data pipeline and metrics as example.py, so the numbers are directly
comparable to a Reckoner run.

The data split is deterministic given the seed, so as long as you use the same
--dataset / seed / config as your Reckoner run, this evaluates on the SAME test
set (you do NOT need to re-run Reckoner to compare).

Usage:
    python baseline.py --dataset compas
    python baseline.py --dataset compas --model mlp --hidden_size 64
    python baseline.py --dataset new_adult --model logistic --epochs 100 --lr 0.0005
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from simple_mlp.trainer import Trainer


# --------------------------------------------------------------------------
# Models (same definitions as example.py)
# --------------------------------------------------------------------------
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def build_model(name, input_size, hidden_size):
    name = name.lower()
    if name in ('logistic', 'lr', 'logreg'):
        return LogisticRegression(input_size)
    if name in ('mlp', 'simplemlp'):
        return SimpleMLP(input_size, hidden_size)
    raise ValueError("Unknown --model '{}'. Use 'logistic' or 'mlp'.".format(name))


# --------------------------------------------------------------------------
# CLI (mirrors example.py; unset overrides fall back to the YAML config)
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run the baseline experiment (no Reckoner).")
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--dataset', type=str, default='compas')
    p.add_argument('--model', type=str, default='logistic',
                   help="Baseline model: 'logistic' or 'mlp'.")
    p.add_argument('--hidden_size', type=int, default=64,
                   help="Hidden size for --model mlp.")

    p.add_argument('--lr', '--learning_rate', dest='learning_rate', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--epochs', '--epoches', dest='epochs', type=int, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--train_ratio', type=float, default=None)
    p.add_argument('--valid_ratio', type=float, default=None)
    p.add_argument('--predict_threshold', type=float, default=None)
    p.add_argument('--quiet', dest='verbose', action='store_false', default=None)
    return p.parse_args()


def build_config(args):
    with open(args.config, 'r') as fh:
        config = yaml.safe_load(fh)
    if args.dataset not in config.get('datasets', {}):
        raise ValueError("Dataset '{}' not found. Available: {}".format(
            args.dataset, list(config.get('datasets', {}).keys())))

    training = dict(config['training'])
    dataset = dict(config['datasets'][args.dataset])

    for key in ['learning_rate', 'batch_size', 'epochs', 'device', 'seed',
                'train_ratio', 'valid_ratio', 'predict_threshold', 'verbose']:
        value = getattr(args, key, None)
        if value is not None:
            training[key] = value
    return training, dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    training, dataset = build_config(args)

    print("BASELINE experiment (no Reckoner framework)")
    print("=" * 50)
    print("Dataset profile : {}".format(args.dataset))
    print("Baseline model  : {}".format(args.model))
    print("Hyperparameters : {}".format(training))

    set_seed(training['seed'])
    device = torch.device(training['device'])

    # --- Load data (identical pipeline to example.py) ---
    X, y, meta = Trainer.load_data(
        dataset['data_name'], dataset['predict_attr'],
        numeric_columns=dataset.get('numeric_columns'),
        categorical_columns=dataset.get('categorical_columns'),
        multihot_columns=dataset.get('multihot_columns'),
        hash_columns=dataset.get('hash_columns'),
        hash_n_features=dataset.get('hash_n_features'),
        sensitive_exclude_columns=dataset.get('sensitive_exclude_columns'),
        sensitive_label_column=dataset.get('sensitive_label_column'),
        race_list=dataset.get('race_list'),
        privileged_race=dataset.get('privileged_race'),
        multivalue_separator=dataset.get('multivalue_separator', ';'),
        match_method=dataset.get('match_method', 'fuzzy'),
        match_threshold=dataset.get('match_threshold', 0.6),
        query_str=dataset.get('query_str'),
    )
    print("Data shape:", X.shape, y.shape)

    input_size = X.shape[1] - 1   # last column is sensitive_info
    print("Detected input_size: {}".format(input_size))

    # --- Same deterministic split/scaling as the Reckoner run ---
    train_set, valid_set, test_set, _ = Trainer.load_train_test_valid(
        X, y, 'all', training['train_ratio'], training['valid_ratio'],
        training['seed'], numeric_columns=meta['numeric_columns'])

    X_train, y_train = train_set.attrs, train_set.labels
    X_val, y_val = valid_set.attrs, valid_set.labels
    X_test, y_test = test_set.attrs, test_set.labels

    # --- Plain supervised training with early stopping on val loss ---
    model = build_model(args.model, input_size, args.hidden_size).to(device)
    print("Number of model parameters: {}".format(sum(p.numel() for p in model.parameters())))
    optimizer = optim.Adam(model.parameters(), lr=training['learning_rate'])
    criterion = nn.BCELoss()

    Xtr = torch.FloatTensor(X_train).to(device)
    ytr = torch.FloatTensor(y_train).to(device)
    Xvl = torch.FloatTensor(X_val).to(device)
    yvl = torch.FloatTensor(y_val).to(device)

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=training['batch_size'], shuffle=True)

    print("\nStarting baseline training...")
    best_val, best_state = float('inf'), None
    for epoch in range(training['epochs']):
        model.train()
        total = 0.0
        for bx, by in train_loader:
            out = model(bx[:, :-1])                 # model never sees sensitive_info
            loss = criterion(out, by.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xvl[:, :-1]), yvl.unsqueeze(1)).item()
        if val_loss < best_val:                     # keep best-on-val model
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if training['verbose'] and epoch % 10 == 0:
            print("Epoch {:3d}: Train Loss = {:.4f}, Val Loss = {:.4f}".format(
                epoch, total / len(train_loader), val_loss))

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Predict on the test set (NO noise added) ---
    print("\nPerforming prediction...")
    model.eval()
    Xte = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        probs = model(Xte[:, :-1]).cpu().numpy().flatten()
    y_pred = (probs > training['predict_threshold']).astype(int)
    y_true = np.asarray(y_test).astype(int).flatten()
    sensitives = np.asarray(X_test)[:, -1]

    # --- Performance (all test rows) ---
    f1 = f1_score(y_true, y_pred, average='binary')
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print("\n===== BASELINE performance analysis (all {} test rows) =====".format(len(y_true)))
    print("F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(f1, precision, recall))

    # --- Fairness (reuse the framework's group_comp; only known-race rows) ---
    results = pd.DataFrame({'predicted_labels': y_pred,
                            'true_labels': y_true,
                            'sensitive_info': sensitives})
    known = results['sensitive_info'].isin([0, 1])
    print("\n===== BASELINE fairness analysis ({} rows; {} excluded for unknown race) =====".format(
        int(known.sum()), int((~known).sum())))
    if int(known.sum()) == 0:
        print("No rows with a known sensitive attribute; skipping fairness analysis.")
    else:
        Trainer.group_comp(None, results, 'sensitive_info', 1)

    print("\nBaseline experiment completed!")
    print("Test F1 score: {:.4f}".format(f1))


if __name__ == "__main__":
    main()
