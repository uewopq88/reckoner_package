"""
Entry point for the QPS fairness model (Reckoner-based recidivism predictor).

Hyperparameters and dataset profiles are read from a YAML config file
(default: config.yaml). Any training hyperparameter can be overridden on the
command line, e.g.:

    python example.py --dataset compas
    python example.py --dataset compas --lr 0.0005 --epochs 200 --run_id 3
    python example.py --config my_config.yaml --dataset new_adult

The main prediction model is logistic regression (task requirement); its
input_size is detected automatically from the loaded data.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml

from simple_mlp.trainer import Trainer
from simple_mlp.noisemodel import NoiseGenerator


# Define model
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


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Reckoner-based fairness model (logistic regression).")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML config file.')
    parser.add_argument('--dataset', type=str, default='compas',
                        help="Dataset profile to use (a key under 'datasets' in the config).")

    # Training hyperparameter overrides. Defaults are None so that, when not
    # passed, the value from the config file is used instead.
    parser.add_argument('--lr', '--learning_rate', dest='learning_rate',
                        type=float, default=None, help='Adam learning rate.')
    parser.add_argument('--batch_size', type=int, default=None, help='Mini-batch size.')
    parser.add_argument('--epochs', '--epoches', dest='epochs',
                        type=int, default=None, help='Max epochs per training stage.')
    parser.add_argument('--device', type=str, default=None, help="'cpu' or 'cuda'.")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Directory where intermediate models are saved.')
    parser.add_argument('--signiture', type=str, default=None,
                        help='Tag added to saved model file names.')
    parser.add_argument('--run_id', type=int, default=None,
                        help='Unique id added to saved model file names.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for the data splits.')
    parser.add_argument('--train_ratio', type=float, default=None, help='Train split ratio.')
    parser.add_argument('--valid_ratio', type=float, default=None, help='Valid split ratio.')
    parser.add_argument('--c_threshold', type=float, default=None,
                        help='High/low confidence identification threshold.')
    parser.add_argument('--ema_momentum', type=float, default=None,
                        help='EMA weight for the high generator during refinement.')
    parser.add_argument('--pseudo_epochs', type=int, default=None,
                        help='Pseudo-learning epochs for the low generator.')
    parser.add_argument('--predict_threshold', type=float, default=None,
                        help='Classification probability threshold.')
    parser.add_argument('--noise_hidden_size', type=int, default=None,
                        help='Hidden size of the learnable NoiseGenerator.')
    parser.add_argument('--quiet', dest='verbose', action='store_false', default=None,
                        help='Disable per-epoch training logs.')
    return parser.parse_args()


def build_config(args):
    """Load the YAML config and apply any command-line overrides."""
    with open(args.config, 'r') as fh:
        config = yaml.safe_load(fh)

    if args.dataset not in config.get('datasets', {}):
        raise ValueError(
            "Dataset '{}' not found in {}. Available: {}".format(
                args.dataset, args.config, list(config.get('datasets', {}).keys())))

    training = dict(config['training'])
    dataset = dict(config['datasets'][args.dataset])

    # Override training hyperparameters with any explicitly-passed CLI values.
    overridable = [
        'learning_rate', 'batch_size', 'epochs', 'device', 'model_path',
        'signiture', 'run_id', 'seed', 'train_ratio', 'valid_ratio',
        'c_threshold', 'ema_momentum', 'pseudo_epochs', 'predict_threshold',
        'noise_hidden_size', 'verbose',
    ]
    for key in overridable:
        value = getattr(args, key, None)
        if value is not None:
            training[key] = value

    return training, dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    training, dataset = build_config(args)

    print("Training the fairness model (logistic regression)")
    print("=" * 50)
    print("Dataset profile: {}".format(args.dataset))
    print("Hyperparameters: {}".format(training))

    set_seed(training['seed'])

    # Load data first so we can detect the input dimension automatically.
    X, y, meta = Trainer.load_data(
        dataset['data_name'],
        dataset['predict_attr'],
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

    # The last column is the sensitive attribute; the model sees X[:, :-1].
    input_size = X.shape[1] - 1
    print("Detected input_size: {}".format(input_size))

    # Create models
    model = LogisticRegression(input_size=input_size)
    low_generator = LogisticRegression(input_size=input_size)
    noise_model = NoiseGenerator(
        in_size=input_size,
        hidden_size=training['noise_hidden_size'],
        output_size=input_size,
    )
    print("Number of model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    # Create trainer
    trainer = Trainer(
        high_generator=model,
        low_generator=low_generator,
        noise_model=noise_model,
        learning_rate=training['learning_rate'],
        batch_size=training['batch_size'],
        epoches=training['epochs'],
        device=training['device'],
        verbose=training['verbose'],
        model_path=training['model_path'],
        signiture=training['signiture'],
        run_id=training['run_id'],
        seed=training['seed'],
        c_threshold=training['c_threshold'],
        ema_momentum=training['ema_momentum'],
        pseudo_epochs=training['pseudo_epochs'],
        predict_threshold=training['predict_threshold'],
    )

    seed = training['seed']
    train_ratio = training['train_ratio']
    valid_ratio = training['valid_ratio']

    train_set, valid_set, test_set, data_size = Trainer.load_train_test_valid(
        X, y, 'all', train_ratio, valid_ratio, seed,
        numeric_columns=meta['numeric_columns'])

    # Stratification
    print("\nStarting stratification...")
    X_train = train_set.attrs
    y_train = train_set.labels
    X_val = valid_set.attrs
    y_val = valid_set.labels
    X_test = test_set.attrs
    y_test = test_set.labels
    hign_id, low_id = trainer.identification(X_train, y_train)
    high_train = X_train[hign_id]
    high_gt = y_train[hign_id]
    low_train = X_train[low_id]
    low_gt = y_train[low_id]
    h_train_set, h_valid_set, _, _ = Trainer.load_train_test_valid(
        high_train, high_gt, 'highConf', 0.9, 0.1, seed)
    l_train_set, l_valid_set, _, _ = Trainer.load_train_test_valid(
        low_train, low_gt, 'lowConf', 0.9, 0.1, seed)

    # Cold start training
    print("\nStarting cold-start training...")
    high_train = h_train_set.attrs
    high_train_labels = h_train_set.labels
    high_valid = h_valid_set.attrs
    high_valid_labels = h_valid_set.labels
    f = 'high_generator'
    high_history = trainer.cold_start(
        high_train, high_train_labels, high_valid, high_valid_labels, f)
    low_train = l_train_set.attrs
    low_train_labels = l_train_set.labels
    low_valid = l_valid_set.attrs
    low_valid_labels = l_valid_set.labels
    f = 'low_generator'
    low_history = trainer.cold_start(
        low_train, low_train_labels, low_valid, low_valid_labels, f)

    # Training
    print("\nStarting refinement training...")  # always train on high generator
    result_history = trainer.refinement(X_train, y_train, X_val, y_val)

    # Prediction
    print("\nPerforming prediction...")
    predictions = trainer.predict(X_test)
    result = trainer.score(X_test, y_test)
    print(f"Test F1 score: {result:.4f}")

    print("\nTest completed!")


if __name__ == "__main__":
    main()
