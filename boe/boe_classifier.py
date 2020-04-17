from sklearn import neural_network
import numpy as np
import evaluation_metrices
import boe


def train(events_list, labels):
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(1000, 1000,), activation='relu',
                                       solver='adam', alpha=1e-4, learning_rate_init=0.001,
                                       early_stopping=True)  # Early stopping is not crucial here
    print("creating feature vec")
    training_data = boe.create_data_embedding(events_list)
    print("start training")
    clf.fit(training_data, labels)
    print("done training")
    return clf


def test(events_list, labels, clf):
    test_data = boe.create_data_embedding(events_list)
    predicted = np.round(clf.predict(test_data))
    evaluation_metrices.evaluate_prediction(predicted, labels)

