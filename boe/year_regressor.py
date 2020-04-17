import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import evaluation_metrices
import boe

net = nn.Sequential(nn.Linear(432, 32, bias=True),
                    nn.ReLU(),
                    nn.Linear(32, 1))


def train_regression(events_list, labels):
    print("creating feature vec")
    training_data = boe.create_data_embedding(events_list)
    print("start training")
    labels = labels.reshape((5446, 1))

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.L1Loss()
    for epoch in range(10000):
        X, Y = Variable(torch.Tensor(training_data), requires_grad=True), \
               Variable(torch.Tensor(labels), requires_grad=False)
        optimizer.zero_grad()
        y_pred = net(X)
        output = criterion(y_pred, Y)
        output.backward()
        optimizer.step()
        if output < 0.00001:
            break
        if epoch % 500 == 0:
            print("Epoch {} - loss: {}".format(epoch, output))


def test_regression(events_list, labels):
    test_data = boe.create_data_embedding(events_list)

    with torch.no_grad():
        predicted = net(Variable(torch.Tensor(test_data)))
        predicted = np.round(predicted.cpu().detach().numpy().flatten())
        evaluation_metrices.evaluate_prediction(predicted, labels)






