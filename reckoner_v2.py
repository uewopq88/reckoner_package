"""
Reckoner v2 -- an improved Trainer that fixes three issues in the original
refinement stage, WITHOUT modifying simple_mlp/trainer.py.

It is a drop-in subclass of ``Trainer``: everything else (data loading,
identification, cold-start, prediction, scoring, fairness) is inherited
unchanged, so results stay directly comparable to the original and to baseline.

What changed vs the original Trainer
------------------------------------
1. Noise now enters the loss (the noise model is actually trained).
   Original: the loss was computed on the CLEAN input, so the noise model's
   gradient was always None and it stayed at its random initialisation -- yet
   ``predict`` added that random noise at test time, destroying test F1.
   v2: the supervised loss is computed on the SAME perturbed input that
   ``predict`` uses, so (a) the noise model receives a gradient and (b) train
   and test see the same condition.

2. EMA refinement is applied ONCE PER EPOCH instead of once per batch.
   Original: ``high = low*(1-ema) + high*ema`` ran every batch, so the
   per-epoch strength was ``ema^(num_batches)`` -- it scaled with dataset size.
   That is why the same ema_momentum was stable on COMPAS / small subsets but
   made the loss diverge on the full (larger) dataset.
   v2: the low generator is pseudo-trained on the whole epoch's data and blended
   in a SINGLE time per epoch, so ema_momentum no longer depends on dataset size
   and transfers across datasets.

3. Validation during refinement is done on the perturbed input too, so the
   best model chosen by early stopping matches the (noisy) test condition.

IMPORTANT -- ema_momentum semantics changed
--------------------------------------------
It is now the "keep fraction per EPOCH", not per batch. ``ema=0.999`` here means
only 0.1% of the low generator is blended in per epoch, which is very weak.
Re-tune from smaller values, e.g. try ema_momentum in {0.9, 0.95, 0.99}.

Usage
-----
See the bottom of this file and the note printed by ``README_example_change()``.
"""

import torch

from simple_mlp.trainer import Trainer


class ReckonerV2(Trainer):
    """Improved Reckoner trainer (see module docstring). Same constructor as
    ``Trainer`` -- construct it exactly the way example.py constructs Trainer."""

    def refinement(self, X_train, y_train, X_val, y_val):
        # Flag so _validate_epoch knows to perturb the validation input during
        # the refinement phase only (cold-start still validates on clean input).
        self._refining = True
        try:
            return super().refinement(X_train, y_train, X_val, y_val)
        finally:
            self._refining = False

    def _train_epoch_with_noise(self, train_loader, model):
        """One refinement epoch.

        Per batch: ordinary supervised training on the PERTURBED input, so both
        the high generator and the noise model get a gradient (fix #1).
        Once at the end of the epoch: pseudo-learning + a single EMA blend
        (fix #2, decoupled from the number of batches).
        """
        self.noise_model.train()
        model.train()
        optimizer = self.high_generator_optimizer

        total_loss = 0.0
        num_batches = 0
        for batch_X, batch_y in train_loader:
            x = batch_X[:, :-1]
            y = batch_y.unsqueeze(1)

            optimizer.zero_grad()
            self.noise_optimizer.zero_grad()

            perturbed = x + self.noise_model(torch.randn_like(x))
            outputs = model(perturbed)
            loss = self.criterion(outputs, y)

            loss.backward()
            optimizer.step()
            self.noise_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Once per epoch: refine the high generator via the low generator.
        self._ema_refine_epoch(train_loader, model)

        return total_loss / num_batches

    def _ema_refine_epoch(self, train_loader, model):
        """Pseudo-train the low generator on the whole epoch's perturbed data,
        then blend it into the high generator a SINGLE time (per epoch)."""
        model.eval()
        self.noise_model.eval()

        inputs, targets = [], []
        with torch.no_grad():
            for batch_X, _ in train_loader:
                x = batch_X[:, :-1]
                perturbed = x + self.noise_model(torch.randn_like(x))
                inputs.append(perturbed)
                targets.append(model(perturbed))
        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)

        tmp_low_path = self.psudo_learning(self.low_generator_path, targets, inputs)
        low_dict = torch.load(tmp_low_path).state_dict()
        with torch.no_grad():
            for key, param in model.state_dict().items():
                if key in low_dict:
                    param.mul_(self.ema_momentum).add_(
                        low_dict[key], alpha=1 - self.ema_momentum)

        model.train()
        self.noise_model.train()

    def _validate_epoch(self, val_loader, model):
        """Validate one epoch. During refinement the validation input is
        perturbed the same way ``predict`` perturbs the test input (fix #3);
        during cold-start it stays clean."""
        model.eval()
        self.noise_model.eval()
        refining = getattr(self, '_refining', False)

        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                x = batch_X[:, :-1]
                if refining:
                    x = x + self.noise_model(torch.randn_like(x))
                outputs = model(x)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
