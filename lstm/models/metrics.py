from typing import Optional
from overrides import overrides

from allennlp.training.metrics.metric import Metric

import torch
import numpy as np
import scipy
import math


class AccumulatedResultScore(Metric):
    """
    This ``Metric`` serves as an interface to accumulate predicted and true model outputs, for use in other metrics
    that need that information. """
    def __init__(self) -> None:
        self._predicted = []
        self._true = []

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor):
        predictions, gold_labels = self.unwrap_to_tensors(predictions, gold_labels)
        self._predicted.extend(predictions.tolist())
        self._true.extend(gold_labels.tolist())

    def get_metric(self, reset: bool = False):
        """ This should be overriden in other metrics. """
        raise NotImplementedError

    def reset(self):
        self._predicted = []
        self._true = []


@Metric.register("kendall_tau_score")
class KendallTauScore(AccumulatedResultScore):
    @overrides
    def get_metric(self, reset: bool = False):
        tau = scipy.stats.kendalltau(self._predicted, self._true)[0]
        if reset:
            self.reset()

        if math.isnan(tau):
            return 0.0

        return tau


@Metric.register("percent_under_year")
class PercentUnderYearRangeScore(AccumulatedResultScore):
    def __init__(self, year_range, year_length=1) -> None:
        super().__init__()

        self._boundary = year_range * year_length

    @overrides
    def get_metric(self, reset: bool = False):
        correct = 0
        total = len(self._predicted)

        for p, t in zip(self._predicted, self._true):
            if p > t - self._boundary and p < t + self._boundary:
                correct += 1

        if reset:
            self.reset()

        return correct / total


class ExactMatchScore(AccumulatedResultScore):
    @overrides
    def get_metric(self, reset: bool = False):
        score = np.sum(np.round(self._predicted) == np.round(self._true)) / len(self._predicted)
        if reset:
            self.reset()

        return score


@Metric.register("mean_squared_error")
class MeanSquaredError(Metric):
    """
    This ``Metric`` calculates the mean squared error (MSE) between two tensors.
    """
    def __init__(self) -> None:
        self._squared_error = 0.0
        self._total_count = 0.0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        squared_errors = (predictions - gold_labels)**2
        if mask is not None:
            squared_errors *= mask
            self._total_count += torch.sum(mask)
        else:
            self._total_count += gold_labels.numel()
        self._squared_error += torch.sum(squared_errors)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated mean squared error.
        """
        mean_squared_error = float(self._squared_error) / float(self._total_count)
        if reset:
            self.reset()
        return mean_squared_error

    @overrides
    def reset(self):
        self._squared_error = 0.0
        self._total_count = 0.0
