from abc import ABC, abstractmethod

import torch


class ClassificationMetric(ABC):
    def __init__(self, threshold=None):
        self._threshold = threshold

    @abstractmethod
    def __call__(self, pred: torch.tensor, actua: torch.tensor):
        pass

    @abstractmethod
    def result(self, validation=False):
        pass


class Accuracy(ClassificationMetric):

    name = "accuracy"

    def __init__(self, threshold=None):
        super().__init__(threshold)
        self._batch_correct_counts = []
        self._batch_sizes = []
        self._batch_wise_metric = []


    def __call__(self, pred: torch.tensor, actual: torch.tensor):
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
        correct = correct_matches.sum()
        self._batch_sizes.extend([total_samples])
        self._batch_correct_counts.extend([correct])
        self._batch_wise_metric.extend([correct * 100 / total_samples])

    def result(self, batched=False):
        if batched:
            return sum(self._batch_correct_counts) * 100 / sum(self._batch_sizes)
        else:
            return sum(self._batch_wise_metric) / len(self._batch_wise_metric)
