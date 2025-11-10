"""
Test Simple MLP Package
"""

import torch
import torch.nn as nn
import numpy as np
from simple_mlp import MLPTrainer
from sklearn.model_selection import train_test_split
import os
from model import NoiseGenerator

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

def main():
    print("Testing Simple MLP Package")
    print("=" * 30)

    # Create models
    model = SimpleMLP(input_size=442, hidden_size=128)
    low_generator = SimpleMLP(input_size=442, hidden_size=128)
    noise_model = NoiseGenerator(in_size=442, hidden_size=128, output_size=442)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create trainer
    trainer = MLPTrainer(
        high_generator=model,
        low_generator=low_generator,
        noise_model=noise_model,
        learning_rate=0.001,
        batch_size=16,
        epoches=50,
        device='cpu',
        verbose=True,
        model_path='temp_models/',
        signiture='test',
        run_id=1
    )

    data_name = "compas-scores-two-years"
    sensitive = 'race'
    predict_attr = "is_recid"
    query_str = "race in ('African-American','Caucasian')"
    intertsted_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                    'age',
                    'c_charge_degree',
                    'c_charge_desc',
                    'age_cat',
                    'sex', 'race', 'is_recid']
    use_FeatureHasher = True
    categorical_features = ['c_charge_desc']
    sensitive_mapping = {'race_African-American': 0,
                        'race_Caucasian': 1}

    X, y = trainer.load_data(data_name, sensitive, predict_attr, intertsted_columns,
                  categorical_features, sensitive_mapping, use_FeatureHasher, query_str)

    print(X.shape)
    print(y.shape)
    # X.to_csv('X.csv', index=False)

    train_set, valid_set, test_set, data_size = trainer.load_train_test_valid(X, y, 'all', 0.50, 0.1)

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
    h_train_set, h_valid_set, _, _ = trainer.load_train_test_valid(high_train, high_gt, 'highConf', 0.9, 0.1)
    l_train_set, l_valid_set, _, _ = trainer.load_train_test_valid(low_train, low_gt, 'lowConf', 0.9, 0.1)

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
    print("\nStarting refinement training...") # always train on high generator
    result_history = trainer.refinement(X_train, y_train, X_val, y_val)

    # Prediction
    print("\nPerforming prediction...")
    predictions = trainer.predict(X_test)
    result = trainer.score(X_test, y_test)
    print(f"Validation accuracy: {result:.4f}")

    # # Get model parameters
    # print("\nModel parameters:")
    # params = trainer.get_model_parameters()
    # for name, param in params.items():
    #     print(f"  {name}: {param.shape}")

    print("\nTest completed!")

if __name__ == "__main__":
    main()
