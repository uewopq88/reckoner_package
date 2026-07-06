"""
Simple MLP Trainer
A simple two-layer MLP trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict, Any
from simple_mlp.identification import identf_split
from simple_mlp import preprocess
import os
from simple_mlp.noisemodel import NoiseGenerator
from torch.autograd import Variable
from collections import OrderedDict
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tabulate import tabulate

class Trainer:
    """
    A simple two-layer MLP trainer.

    This trainer accepts externally defined models and only handles the training process.
    """

    def __init__(
        self,
        high_generator: nn.Module,
        low_generator: nn.Module,
        noise_model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epoches: int = 100,
        device: str = 'cpu',
        verbose: bool = True,
        model_path: str = './temp_models',
        signiture: str = 'test',
        run_id: int = 1,
        seed: int = 7,
        c_threshold: float = 0.6,
        ema_momentum: float = 0.999,
        pseudo_epochs: int = 3,
        predict_threshold: float = 0.5
    ):
        """
        Initialize the trainer.

        Args:
            high_generator: externally defined high-confidence PyTorch model
            low_generator: externally defined low-confidence PyTorch model
            noise_model: learnable noise generator
            learning_rate: learning rate
            batch_size: batch size
            epoches: number of training epochs
            device: device ('cpu' or 'cuda')
            verbose: whether to print training logs
            model_path: path to save models
            signiture: tag added to saved model file names
            run_id: unique run id added to saved model file names
            seed: random seed for the train/valid/test splits
            c_threshold: confidence threshold for the high/low identification split
            ema_momentum: EMA weight for the high generator during refinement
            pseudo_epochs: number of pseudo-learning epochs for the low generator
            predict_threshold: probability threshold for binary classification
        """
        self.high_generator = high_generator
        self.low_generator = low_generator
        self.noise_model = noise_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoches = epoches
        self.device = torch.device(device)
        self.verbose = verbose
        self.model_path = model_path
        self.signiture = signiture
        self.run_id = run_id
        self.seed = seed
        self.c_threshold = c_threshold
        self.ema_momentum = ema_momentum
        self.pseudo_epochs = pseudo_epochs
        self.predict_threshold = predict_threshold

        # Move models to the specified device
        self.high_generator.to(self.device)
        self.low_generator.to(self.device)
        self.noise_model.to(self.device)

        # Initialize optimizers and loss functions
        self.high_generator_optimizer = optim.Adam(self.high_generator.parameters(), lr=learning_rate)
        self.low_generator_optimizer = optim.Adam(self.low_generator.parameters(), lr=learning_rate)
        self.noise_optimizer = optim.Adam(self.noise_model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.low_generator_criterion = nn.BCELoss()
        # Training history
        self.train_losses = []
        self.val_losses = []

        self.low_generator_path = None
        self.high_generator_path = None
        self.noise_model_path = None


    def group_comp(self, df_pred,label,privileged_group):

        g1 = privileged_group
        g0 = privileged_group^1

        # Only rows with a known sensitive attribute (0/1) take part in the
        # fairness analysis; unknown rows (sentinel -1) are excluded.
        df_pred = df_pred[df_pred[label].isin([g0, g1])]

        # Privileged group
        df_priv = df_pred[df_pred[label]==g1]
        priv_truth = df_priv['true_labels']
        priv_pred = df_priv['predicted_labels']

        pr1=len([i for i in priv_pred if i==1])/len(priv_pred)
        cm = confusion_matrix(priv_truth,priv_pred,labels=[0,1])
        tn1, fp1, fn1, tp1=cm.ravel()
        g1_results = [ f1_score(priv_truth,priv_pred,average='weighted'), tp1/(tp1+fn1), fp1/(fp1+tn1), pr1]
        print("this is pr1: ", pr1)

        # Non-privileged group
        df_nopriv = df_pred[df_pred[label]==g0]
        nopriv_truth = df_nopriv['true_labels']
        nopriv_pred = df_nopriv['predicted_labels']

        pr0=len([i for i in nopriv_pred if i==1])/len(nopriv_pred)
        cm = confusion_matrix(nopriv_truth,nopriv_pred,labels=[0,1])
        tn0, fp0, fn0, tp0 = cm.ravel()
        g0_results = [f1_score(nopriv_truth,nopriv_pred,average='weighted'), tp0/(tp0+fn0), fp0/(fp0+tn0), pr0]
        print("this is pr0: ", pr0)

        # Print the summary of comparison
        table = [['Group', 'F1', 'TPR', 'FPR', 'PR'], ['Privileged']+g1_results, ['Non-privileged']+g0_results]
        print(tabulate(table, floatfmt='.3f', headers = "firstrow", tablefmt='psql'))

        # eop=tp0/(tp0+fn0)-tp1/(tp1+fn1)
        eodds= abs((tp0/(tp0+fn0)-tp1/(tp1+fn1))*0.5+(fp0/(fp0+tn0)-fp1/(fp1+tn1))*0.5)
        sp = abs(pr0-pr1)
        # print("Equal Opportunity %.4f"%(eop))
        print("Equal Odds %.4f" %(eodds))
        print("Demographic Parity %.4f"%(sp))

    @staticmethod
    def load_data(csv_name, predict_attr,
                  numeric_columns=None, categorical_columns=None,
                  multihot_columns=None, hash_columns=None, hash_n_features=None,
                  sensitive_exclude_columns=None, sensitive_label_column=None,
                  race_list=None, privileged_race=None,
                  multivalue_separator=';', match_method='fuzzy',
                  match_threshold=0.6, query_str=None):
        """
        Load and preprocess a dataset according to the per-column configuration.

        Returns:
            X    : feature DataFrame; the LAST column is 'sensitive_info'
                   (1 = privileged, 0 = non-privileged, -1 = unknown).
            y    : label array.
            meta : dict with 'numeric_columns' (names needing standardisation).
        """
        df = pd.read_csv("./data/{}.csv".format(csv_name))
        if query_str is not None:
            df = df.query(query_str)
        df = df.reset_index(drop=True)

        # Drop rows without a label -- they cannot be used for training.
        n_before = len(df)
        df = df[df[predict_attr].notna()].reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped > 0:
            print("Dropped {} rows with a missing '{}' label.".format(dropped, predict_attr))

        # Guard: sensitive columns must never appear among the feature columns.
        sensitive_exclude_columns = list(sensitive_exclude_columns or [])
        feature_cols = (list(numeric_columns or []) + list(categorical_columns or [])
                        + list(multihot_columns or []) + list(hash_columns or []))
        leaked = [c for c in feature_cols if c in sensitive_exclude_columns]
        if leaked:
            raise ValueError(
                "Sensitive columns {} must not be used as features.".format(leaked))

        y = df[predict_attr].astype(float).values

        # Build the feature matrix from the per-type column lists.
        X, numeric_out = preprocess.process_features(
            df,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            multihot_columns=multihot_columns,
            hash_columns=hash_columns,
            hash_n_features=hash_n_features,
            sep=multivalue_separator,
        )
        X = X.sort_index(axis=1)

        # Derive the fairness group label (privileged / non-privileged / unknown).
        if sensitive_label_column is not None:
            sensitive_info = preprocess.build_sensitive_info(
                df[sensitive_label_column], race_list, privileged_race,
                sep=multivalue_separator, method=match_method,
                threshold=match_threshold)
        else:
            sensitive_info = np.full(len(df), preprocess.UNKNOWN_SENSITIVE)
        X['sensitive_info'] = sensitive_info

        # Ensure 'sensitive_info' is the LAST column (model uses X[:, :-1]).
        ordered = [c for c in X.columns if c != 'sensitive_info'] + ['sensitive_info']
        X = X[ordered].astype(float)

        n_priv = int((sensitive_info == 1).sum())
        n_nonpriv = int((sensitive_info == 0).sum())
        n_unknown = int((sensitive_info == preprocess.UNKNOWN_SENSITIVE).sum())
        print("Sensitive groups -> privileged: {}, non-privileged: {}, unknown: {}".format(
            n_priv, n_nonpriv, n_unknown))

        return X, y, {'numeric_columns': numeric_out}

    @staticmethod
    def load_train_test_valid(X, y, f, train_ratio, valid_ratio, seed=7, numeric_columns=None):
        class A_C(Dataset):
            def __init__(self, attrs, labels):
                self.attrs = attrs
                self.labels = labels

            def __getitem__(self, idx):
                return [self.attrs[idx], self.labels[idx]]

            def __len__(self):
                return len(self.labels)

        data_size = len(X)
        true_test_ratio = 1 - (train_ratio + valid_ratio)
        rest_data = data_size * (1 - true_test_ratio)
        true_valid_ratio = (rest_data - data_size * train_ratio) / rest_data

        train, valid, train_labels, valid_labels = train_test_split(X, y, test_size=true_valid_ratio, random_state=seed)
        if f != 'highConf' and f != 'lowConf':
            # Split BEFORE scaling so the scaler only sees the training data.
            # Fitting StandardScaler on the full X would leak the mean/variance
            # of the validation and test splits into training (data leakage).
            train, test, train_labels, test_labels = train_test_split(X, y, test_size=true_test_ratio, random_state=seed)
            train, valid, train_labels, valid_labels = train_test_split(train, train_labels, test_size=true_valid_ratio, random_state=seed)
            # Standardise ONLY the numeric columns (categorical / multi-hot /
            # hashed / sensitive columns are left as-is). Fit on the training
            # split, then apply the same transform to validation and test.
            if numeric_columns:
                ct = ColumnTransformer([('num', StandardScaler(), list(numeric_columns))],
                                       remainder='passthrough')
                train = ct.fit_transform(train)
                valid = ct.transform(valid)
                test = ct.transform(test)
            else:
                train = np.asarray(train, dtype=float)
                valid = np.asarray(valid, dtype=float)
                test = np.asarray(test, dtype=float)
            print('after scaling: ', train.shape)

        train_set = A_C(train, train_labels)
        print("len of train labels: ", train_set.__len__())
        valid_set = A_C(valid, valid_labels)
        print("len of valid labels: ", valid_set.__len__())
        train_size = len(train)
        valid_size = len(valid)
        test_size = 0
        test_set = []
        if f != 'highConf' and f != 'lowConf':
            test_set = A_C(test, test_labels)
            test_size = len(test)
            print("len of test labels: ", test_set.__len__())

        print("The size of training set is: {}".format(train_size))
        print("The size of testing set is: {}".format(test_size))
        print("The size of valid set is: {}".format(valid_size))

        data_size = [train_size, test_size, valid_size]
        return train_set, valid_set, test_set, data_size


    def identification(self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identification
        """
        return identf_split(X_train, y_train, self.c_threshold)

    def cold_start(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        f: str
    ) -> Dict[str, Any]:
        """
        Cold-start training
        """
        if f == 'low_generator':
            self.low_generator_path = os.path.join(
                self.model_path, "model_{}_low_{}.pt".format(self.signiture, self.run_id))
            return self.train(X_train, y_train, X_val, y_val, self.low_generator, self.low_generator_path, False)
        elif f == 'high_generator':
            self.high_generator_path = os.path.join(
                self.model_path, "model_{}_high_{}.pt".format(self.signiture, self.run_id))
            return self.train(X_train, y_train, X_val, y_val, self.high_generator, self.high_generator_path, False)
        else:
            raise ValueError("Invalid model type: {}".format(f))


    def refinement(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Model refinement training
        """
        return self.train(X_train, y_train, X_val, y_val, self.high_generator, self.high_generator_path, True)


    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model: nn.Module,
        model_path: str = None,
        with_noise: bool = False
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            X_train: training features
            y_train: training labels
            X_val: validation features
            y_val: validation labels
            model: model to train
            model_path: path to save model
            with_noise: whether to use noise

        Returns:
            A dictionary of training history
        """

        if with_noise:
            self.noise_model_path = os.path.join(
                self.model_path, "model_{}_noise_{}.pt".format(self.signiture, self.run_id))
            print("Temp location for noise models: {}".format(self.noise_model_path))
            os.makedirs(os.path.dirname(self.noise_model_path), exist_ok=True)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation data
        val_loader = None
        min_valid_loss = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            min_valid_loss = float('inf')

        # Training loop
        for epoch in range(self.epoches):
            # Training phase
            if with_noise:
                train_loss = self._train_epoch_with_noise(train_loader, model)
            else:
                train_loss = self._train_epoch(train_loader, model)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, model)
                self.val_losses.append(val_loss)

                if val_loss < min_valid_loss:
                    min_valid_loss = val_loss
                    torch.save(model, model_path)
                    if with_noise and self.noise_model_path is not None:
                        torch.save(self.noise_model, self.noise_model_path)
                    print("Found new best model, saving to disk...")
                    print("\n")

            # Print progress
            if self.verbose and epoch % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'model_path': self.model_path
        }

    def psudo_learning(self, low_generator_path, psudo_predicted, batch_input):
        epochs = self.pseudo_epochs
        low_model = torch.load(low_generator_path)

        model_path = os.path.join(
            self.model_path, "model_{}_lowtmp_{}.pt".format(self.signiture, self.run_id))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        parameters = filter(lambda p: p.requires_grad, low_model.parameters())
        optimizer = optim.Adam(parameters, lr=self.learning_rate, weight_decay = 0)

        for e in range(epochs):
            low_model.train()
            low_model.zero_grad()
            # Forward pass
            outputs = low_model(batch_input)
            loss_low = self.low_generator_criterion(outputs, psudo_predicted)

            # Backward propagation
            optimizer.zero_grad()
            loss_low.backward()
            optimizer.step()

        torch.save(low_model, model_path)
        return model_path

    def _train_epoch(self, train_loader: DataLoader, model: nn.Module) -> float:
        """Train one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Select the optimizer according to the model
        if model is self.low_generator:
            optimizer = self.low_generator_optimizer
        elif model is self.high_generator:
            optimizer = self.high_generator_optimizer
        else:
            raise ValueError("Unknown model type")

        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X[:, :-1])
            loss = self.criterion(outputs, batch_y.unsqueeze(1))

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _train_epoch_with_noise(self, train_loader: DataLoader, model: nn.Module) -> float:
        """Train one epoch (with noise, used in refinement phase)"""

        self.noise_model.train()
        model.train()
        total_loss = 0.0
        num_batches = 0

        optimizer = self.high_generator_optimizer


        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            self.noise_optimizer.zero_grad()
            # Forward pass
            noise = Variable(torch.randn(batch_X[:, :-1].size()).float(), requires_grad=False).to(self.device)
            generated_noise = self.noise_model(noise)
            new_batch_input = batch_X[:, :-1] + generated_noise

            # First calculation for pseudo training
            outputs = model(new_batch_input)
            tmp_low_generator_path = self.psudo_learning(self.low_generator_path, outputs.detach(), new_batch_input.detach())

            # Refinement core
            low_generator = torch.load(tmp_low_generator_path)
            lowConf_dict = low_generator.state_dict()
            new_highConf_dict = OrderedDict()
            for key, value in model.state_dict().items():
                if key in lowConf_dict.keys():
                    new_highConf_dict[key] = (
                        lowConf_dict[key] * (1 - self.ema_momentum) + value * self.ema_momentum
                    )
            model.load_state_dict(new_highConf_dict)
            new_predicted = model(batch_X[:, :-1])
            loss = self.criterion(new_predicted, batch_y.unsqueeze(1))

            # Backward propagation
            loss.backward()
            optimizer.step()
            self.noise_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, val_loader: DataLoader, model: nn.Module) -> float:
        """Validate one epoch"""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X[:, :-1])
                loss = self.criterion(outputs, batch_y.unsqueeze(1))

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediction

        Args:
            X: input features

        Returns:
            prediction results
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        test_iterator = DataLoader(X_tensor, batch_size=128, num_workers=0, shuffle=False)
        best_model = torch.load(self.high_generator_path)
        best_noise_model = torch.load(self.noise_model_path)
        best_model.eval()
        best_noise_model.eval()

        output_test_list = []
        # ground truths
        sensitives_list = []

        with torch.no_grad():
            for batch_X in test_iterator:
                noise = Variable(torch.randn(batch_X[:, :-1].size()).float(), requires_grad=False).to(self.device)
                generated_noise = best_noise_model(noise)
                new_batch_input = batch_X[:, :-1] + generated_noise

                outputs = best_model(new_batch_input)
                sensitives_list.append(batch_X[:, -1])
                predictions = (outputs > self.predict_threshold).float().cpu().numpy()
                output_test_list.append(predictions)

        # Convert lists into numpy arrays and flatten
        return np.concatenate(output_test_list, axis=0).flatten(), np.concatenate(sensitives_list, axis=0).flatten()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute and return the F1 score for a binary classification task

        Args:
            X: input features
            y: true labels (0/1)

        Returns:
            F1 score (float)
        """
        predictions, sensitives_list = self.predict(X)
        results = pd.DataFrame(predictions, columns=['predicted_labels'])
        results['true_labels'] = y
        results['sensitive_info'] = sensitives_list

        # Ensure 1D 0/1 arrays
        y_true = np.asarray(y).astype(int).flatten()
        y_pred = np.asarray(predictions).astype(int).flatten()

        # Compute confusion matrix elements
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        # Compute precision, recall, and derive F1
        precision_den = tp + fp
        recall_den = tp + fn
        precision = (tp / precision_den) if precision_den > 0 else 0.0
        recall = (tp / recall_den) if recall_den > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # ---- Performance analysis: computed over ALL test rows ----
        print("\n===== Performance analysis (all {} test rows) =====".format(len(y_true)))
        print("F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(f1, precision, recall))

        # ---- Fairness analysis: only rows with a known sensitive attribute ----
        known = results['sensitive_info'].isin([0, 1])
        n_known = int(known.sum())
        n_excluded = int((~known).sum())
        print("\n===== Fairness analysis ({} rows; {} excluded for unknown race) =====".format(
            n_known, n_excluded))
        if n_known == 0:
            print("No rows with a known sensitive attribute; skipping fairness analysis.")
        else:
            self.group_comp(results, 'sensitive_info', 1)

        return float(f1)
