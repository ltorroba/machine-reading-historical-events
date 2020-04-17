import numpy as np
from scipy.stats import kendalltau


def evaluate_prediction(predicted, true_labels):
    num_events = len(predicted)
    print("tau:", kendalltau(predicted, true_labels)[0])

    diff = np.abs(true_labels - predicted)

    print("mean: ", np.mean(diff))

    diff = diff[diff <= 50]
    print("under 50: ", 100 * diff.size / num_events)

    diff = diff[diff <= 20]
    print("under 20: ", 100 * diff.size / num_events)

    diff = diff[diff < 1]
    print("exact: ", 100 * diff.size / num_events)