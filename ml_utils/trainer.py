import math
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import torch


class Trainer:
    """A trainer responsible for training PyTorch models and recording
    scores and other training stats."""
    def __init__(self, max_epochs=10, device="cpu", cleanup=False, track_metrics=[]):
        self._max_epochs = max_epochs
        self._device = torch.device(device)
        self._cleanup = cleanup
        self._metrics = track_metrics
        self._history = {
            "train": {"loss": []},
            "test": {"loss": []}
        }
        for metric in track_metrics:
            self._history["train"][metric.name] = []
            self._history["test"][metric.name] = []
            
    def prepare(self, model,  optimizer, loss_fn, train_data, val_data=None):
        """Prepares the trainer by keeping, model """
        self._train_data = train_data
        self._val_data = val_data
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._model = self._model.to(self._device)
    
    def _display_progress(self, batch, total_batches, duration):
        """Shows the progress bar on the terminal."""
        percent = batch * 100 / total_batches
        n_equals = int(percent * 0.4)
        n_dots = 40 - n_equals
        prog_str = f"\r[{n_equals*'='}{n_dots*'.'}] - Batch:{batch}/{total_batches} | {percent:.2f}% completed | {duration:.2f}s elapsed."
        sys.stdout.write(prog_str)
        sys.stdout.flush()

    def _display_info(self, val=False):
        metrics = self._history["train"]
        prefix = "Training: "
        if val:
            metrics = self._history["test"]
            prefix = "Validation: "
        metric_strings = [
            f"{m}: {v[-1]}" for m, v in metrics.items()
        ]
        display_string = prefix + " | ".join(metric_strings)
        print(display_string)

    def _fit_epoch(self, verbose=False):
        """Fits model for one epoch of data"""
        start = time.time()
        total_batches = len(self._train_data)
        metric_estimators = [Metrics() for Metrics in self._metrics]
        total_loss = 0
        for batch, (X, y) in enumerate(self._train_data):
            X, y = X.to(self._device), y.to(self._device)
            pred_output = self._model(X).squeeze()            
            loss = self._loss_fn(pred_output, y)
            
            total_loss += loss.item()
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            end = time.time()
            # Accumulating metrics
            for estimator in metric_estimators:
                with torch.no_grad():
                    estimator(pred_output, y)
            if verbose:
                self._display_progress(batch+1, total_batches, end-start)
            if self._device != torch.device("cpu") and self._cleanup:
                del X, y
                torch.cuda.empty_cache()
            
        # Recording history
        train_loss = total_loss / total_batches
        self._history["train"]["loss"].extend([train_loss])
        # Recording metrics
        for estimator in metric_estimators:
            self._history["train"][estimator.name].extend(
                [estimator.result(batched=True)])
        if self._val_data is not None:
            self._estimate_val_metrics(self._val_data)
        if verbose:
            print()
            self._display_info()
            if self._val_data is not None:
                self._display_info(val=True)

    def fit(self, verbose=True):
        """Fits the model to the training data."""
        for epoch in range(1, self._max_epochs+1):
            if verbose:
                print(f"Epoch: {epoch}/{self._max_epochs}")
            self._fit_epoch(verbose)

    def _estimate_val_metrics(self, val_data=None, track=True):
        """Checks loss on validation set rather than."""

        total_loss = 0
        num_batches = len(val_data)
        metric_estimators = [Metrics() for Metrics in self._metrics]
        for (X, y) in val_data:
            with torch.no_grad():
                X, y = X.to(self._device), y.to(self._device)
                pred_output = self._model(X).squeeze()
                loss_val = self._loss_fn(pred_output, y, reduction="sum").item()
                total_loss += loss_val
                for estimator in metric_estimators:
                    estimator(pred_output, y)
        
        avg_loss = total_loss / num_batches
        # Getting the final metrics
        metrics = {"loss": avg_loss}
        for estimator in metric_estimators:
            metrics[estimator.name] = estimator.result()
        if not track:
            return metrics
        for name, val in metrics.items():
            self._history["test"][name].extend([val])

    def validate(self, val_data):
        """Gives validation metrics on a new data."""
        return self._estimate_val_metrics(val_data, track=False)

    def plot_history(self):
        ncols = 2
        nrows = math.ceil(len(self._history["train"]) / ncols)
        _, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)
        metric_names = list(self._history["train"].keys())
        train_metrics = list(self._history["train"].values())
        test_metrics = list(self._history["test"].values())
        epochs = list(range(len(train_metrics[0])))
        current_metric = 0
        for i in range(nrows):
            for j in range(ncols):
                axis = axes[i][j]
                axis.plot(epochs, train_metrics[current_metric], label="train", linestyle="--")
                axis.plot(epochs, test_metrics[current_metric], label="test", color="orange")
                axis.set_title(metric_names[current_metric])
                axis.set_xlim([0, len(epochs)])
                axis.legend()
                current_metric += 1
        plt.tight_layout()