from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class ClassificationMetric(ABC):
    def __init__(self, threshold=None):
        self._threshold = threshold

    @abstractmethod
    def __call__(self, pred: torch.tensor, actua: torch.tensor):
        pass

    @abstractmethod
    def result(self):
        pass


class Accuracy(ClassificationMetric):

    name = "accuracy"

    def __init__(self, threshold=None):
        super().__init__(threshold)
        self._batch_correct_counts = []
        self._batch_sizes = []
        self._batch_wise_metric = []

    def __call__(self, pred: torch.Tensor, actual: torch.Tensor):
        # For multiclass one-hot encoded vectors
        if pred.ndim > 1:
            predicted_labels = pred.argmax(axis=1)
            correct_matches = predicted_labels == actual
        # For binary classification probabilities
        else:
            if self._threshold is None:
                self._threshold = 0.5
            matches = pred > self._threshold
            correct_matches = matches == actual
        
        total_samples = pred.shape[0]
        correct = correct_matches.sum().item()
        self._batch_sizes.extend([total_samples])
        self._batch_correct_counts.extend([correct])
        self._batch_wise_metric.extend([correct * 100 / total_samples])

    def result(self):
        return sum(self._batch_correct_counts) * 100 / sum(self._batch_sizes)


class Perplexity(ClassificationMetric):
    """Defines perplexity metric language modelling."""
    name = "perplexity"

    def __init__(self):
        self._entropies = []
        self._batch_sizes = []

    def __call__(self, pred: torch.Tensor, actual: torch.Tensor):
        ent_val = F.cross_entropy(pred, actual, reduction="sum")
        self._entropies.append(ent_val)
        self._batch_sizes.append(len(pred))
    
    def result(self):
        return torch.exp(sum(self.entropies) / sum(self._batch_sizes))