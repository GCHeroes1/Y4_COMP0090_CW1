# https://pytorch.org/vision/stable/datasets.html
# https://pytorch.org/docs/stable/data.html
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
import torch.linalg
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import math
dtype = torch.float32


def polynomial_fun(weight_vector, scalar_variable_x):
    """
    computes the polynomial with an x variable and given weight vector
    :param weight_vector: weight vector either as a n,1 matrix or 1, n
    :param scalar_variable_x: either a scalar variable or a polynomial mapping
    :return: y value for polynomial given x
    """
    return torch.mm(scalar_variable_x, weight_vector)
    # M = len(weight_vector) - 1
    # x = torch.ones_like(weight_vector) * scalar_variable_x
    # x_features = create_features(x, M)
    # y = torch.mm(x_features, weight_vector)
    # # print(y)
    # # for m_index, m_values in enumerate(weight_vector):
    # #     print(m_index)
    # #     y.item(m_index) = torch.dot(weight_vector[m_index], scalar_variable_x[m_index])
    # # print(y)
    # return y


def fit_polynomial_ls(x_coordinates, t_target_values, ls_degree):
    """
    implementation of least squares, taking set of coordinates and labels and fitting them with least squares
    :param x_coordinates:
    :param t_target_values:
    :param ls_degree:
    :return:
    """
    # the weights are output in backwards order, the lowest degree is the start of the vector
    ones = torch.ones_like(x_coordinates.squeeze(1))
    x_degrees_to_be_added = list()
    x_degrees_to_be_added.append(ones)
    x_degrees_to_be_added.append((x_coordinates.squeeze(1)))
    # x = torch.stack((ones, x_coordinates), 1)
    for i in range(1, ls_degree + 1):
        if i > 1:
            x_degrees_to_be_added.append(x_coordinates.squeeze(1) ** i)
    x_degrees = torch.stack(x_degrees_to_be_added, 1)
    # print(x_degrees)
    # weights = torch.lstsq(t_target_values, x_degrees).solution[:x_degrees.size(1)]

    x_degrees_transpose = torch.transpose(x_degrees, 0, 1)
    comp1 = torch.matmul(x_degrees_transpose, x_degrees)
    comp2 = torch.inverse(comp1)
    comp3 = torch.matmul(x_degrees_transpose, t_target_values)
    weights = torch.matmul(comp2, comp3)

    return weights


class PolyModel(nn.Module):
    """
    Model for polynomial regression
    """
    def __init__(self, degree_=2):
        super(PolyModel, self).__init__()
        self.poly = nn.Linear(degree_, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


# https://github.com/ConsciousML/Polynomial-Regression-from-scratch/blob/master/Polynomial%20Regression.ipynb
def make_features(X, degree):
    """
    function to return Vandermonde matrix
    :param X: data
    :param degree: degree
    :return: Vandermonde matrix
    """
    if len(X.shape) == 1:
        X = X.unsqueeze(1)
    # Concatenate a column of ones to has the bias in X
    ones_col = torch.ones((X.shape[0], 1), dtype=torch.float32)
    X_d = torch.cat([ones_col, X], axis=1)
    for i in range(1, degree):
        X_pow = X.pow(i + 1)
        # If we use the gradient descent method, we need to
        # standardize the features to avoid exploding gradients
        X_d = torch.cat([X_d, X_pow], axis=1)
    return X_d


# def make_features(x, degree):
#     m, _ = x.shape
#
#     stack = []
#     for i in range(degree):
#         stack.append(x.pow(i))
#     x_feature = torch.stack(stack)
#     x_feature = torch.reshape(x_feature, (m, degree))
#     return x_feature


# def make_features(x, degree):
#     '''Builds features i.. a matrix with columns [x,x^2,x^3].'''
#     # x = x.unsqueeze(1)
#     return torch.cat([x ** i for i in range(1, degree + 1)], 1)


def fit_polynomial_sgd(x_coordinates, t_target_values, degree, learning_rate, minibatch_size):
    """
    stochastic gradient fitting
    :param x_coordinates: coordinates
    :param t_target_values: labels
    :param degree: degree
    :param learning_rate: LR
    :param minibatch_size: batch size
    :return: trained model
    """
    # degree = 3
    model = PolyModel(degree_=degree+1)

    features = make_features(x_coordinates, degree=degree)
    dataset = TensorDataset(features, t_target_values)

    training_loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True, num_workers=1)

    # loss and optimiser
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    epochs = 1000
    for epoch in range(epochs):
        running_loss, total = 0.0, 0
        for i, data in enumerate(training_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total += labels.size(0)
        loss = float(running_loss) / float(total)
        if epoch % 50 == 49:
            print(f"loss is {str(loss)} after {str(epoch + 1)} epochs")
    print('Training done.')

    # save trained model
    torch.save(model.state_dict(), 'SGD_model.pt')
    print('Model saved.')
    return model


def test_model(model, testloader):
    """
    function to test the model against the test_loader, as well as calculate mean and std
    :param model:
    :param testloader:
    :return: mean and std
    """
    model.eval()
    diff, total, std, std_count = 0, 0, 0, 0
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        with torch.no_grad():
            outputs = model(inputs)
        pred = torch.max(outputs, 1)[1]

        # temp = (pred - labels)
        diff += (pred - labels).sum()
        total += labels.size(0)

        temp = (pred - labels)
        mean = temp.sum() / labels.size(0)
        var = ((temp-mean)**2).sum() / len(temp)
        std += math.sqrt(var)
        std_count += 1
    model.train()
    mean = diff / total
    std = std / std_count
    return mean, std


if __name__ == '__main__':
    dtype = torch.float32
    # x_coordinates = torch.randint(low=0, high=10, size=(10,), dtype=dtype)
    # t_coordinates = torch.randint(low=0, high=10, size=(10,), dtype=dtype)
    # degree = 2

    # print("resulting weight of LS fit: " + str(fit_polynomial_ls(x_coordinates, t_coordinates, degree)))

    # print("resulting weight of SGD fit: " + str(fit_polynomial_sgd(x_coordinates, t_coordinates, degree, learning_rate, minibatch_size)))
    # data = torch.stack((x_coordinates, t_coordinates), 1)

    DEGREE = 4
    LEARNING_RATE = 1e-11
    MINIBATCH_SIZE = 12

    weight_vector = torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1)  # Add Second Dimension
    # print(weight_vector)

    x_training_coordinates = torch.randint(low=-20, high=20, size=(100,), dtype=dtype).unsqueeze(1)  # ([100, 1])
    x_testing_coordinates = torch.randint(low=-20, high=20, size=(50,), dtype=dtype).unsqueeze(1)  # ([50, 1])
    # print(x_training_coordinates.shape, x_testing_coordinates.shape)  # ([100, 1]) ([50, 1])

    # feature map of x_training_coordinates and x_testing_coordinates
    x_training_map = make_features(x_training_coordinates, 3)  # using the x_training coords
    x_testing_map = make_features(x_testing_coordinates, 3)  # using the x_testing coords
    # print(x_training_map.shape, x_testing_map.shape)  # ([100, 4]) ([50, 4])

    t_training_coordinates_true = polynomial_fun(weight_vector, x_training_map)  # calculate t = f(x_training)
    noise = torch.normal(mean=0.0, std=0.2, size=t_training_coordinates_true.shape, dtype=dtype)
    t_training_coordinates = t_training_coordinates_true + noise
    # print(t_training_coordinates.shape)  # ([100, 1])

    t_testing_coordinates_true = polynomial_fun(weight_vector, x_testing_map)  # calculate t = f(x_testing)
    noise = torch.normal(mean=0.0, std=0.2, size=t_testing_coordinates_true.shape, dtype=dtype)
    t_testing_coordinates = t_testing_coordinates_true + noise
    # print(t_testing_coordinates.shape)  # ([50, 1])
    # print("done sampling")

    LS_optimum_weight_vector = fit_polynomial_ls(x_training_coordinates, t_training_coordinates, DEGREE)  # (5, 1)
    print(f"the LS optimised weights are \n{str(LS_optimum_weight_vector)}")

    # print(LS_optimum_weight_vector.shape)
    x_training_map_new = make_features(x_training_coordinates, DEGREE)
    x_testing_map_new = make_features(x_testing_coordinates, DEGREE)

    y_hat_training = polynomial_fun(LS_optimum_weight_vector, x_training_map_new)
    y_hat_testing = polynomial_fun(LS_optimum_weight_vector, x_testing_map_new)
    # print(y_hat_training.shape, y_hat_testing.shape)

    observed_diff = list()
    # diff between t_training_coordinates and t_training_coordinates_true
    for i, (y_hat, y) in enumerate(zip(t_training_coordinates, t_training_coordinates_true)):
        observed_diff.append(y_hat - y)
    for i, (y_hat, y) in enumerate(zip(t_testing_coordinates, t_testing_coordinates_true)):
        observed_diff.append(y_hat - y)
    print(f" a - Observed data mean difference = {torch.mean(torch.FloatTensor(observed_diff))}")
    print(f" a - Observed data std = {torch.std(torch.FloatTensor(observed_diff))}")

    x_new_training_map = make_features(x_training_coordinates, DEGREE)
    x_new_testing_map = make_features(x_testing_coordinates, DEGREE)
    y_hat_LS_training = polynomial_fun(LS_optimum_weight_vector, x_new_training_map)
    y_hat_LS_testing = polynomial_fun(LS_optimum_weight_vector, x_new_testing_map)

    LS_diff = list()
    # diff between predictions made by LS and t_training_coordinates_true
    for i, (y_hat, y) in enumerate(zip(y_hat_LS_training, t_training_coordinates_true)):
        LS_diff.append(y_hat - y)
    for i, (y_hat, y) in enumerate(zip(y_hat_LS_testing, t_testing_coordinates_true)):
        LS_diff.append(y_hat - y)
    print(f" b - LS data mean difference = {torch.mean(torch.FloatTensor(LS_diff))}")
    print(f" b - LS data std = {torch.std(torch.FloatTensor(LS_diff))}")

    # uncomment for training SGD model, need to change save path though
    SGD_model = fit_polynomial_sgd(x_training_coordinates, t_training_coordinates, DEGREE,
                                   LEARNING_RATE, MINIBATCH_SIZE)

    # model = PolyModel(degree_=DEGREE+1)
    # model.load_state_dict(torch.load('./task1/SGD_model.pt'))
    model = PolyModel(degree_=DEGREE+1)
    model.load_state_dict(torch.load('SGD_model.pt'))

    training_features = make_features(x_training_coordinates, degree=DEGREE)
    testing_features = make_features(x_testing_coordinates, degree=DEGREE)
    training_dataset = TensorDataset(training_features, t_training_coordinates_true)
    testing_dataset = TensorDataset(testing_features, t_testing_coordinates_true)

    mean = list()
    std = list()
    training_loader = DataLoader(training_dataset, batch_size=MINIBATCH_SIZE, shuffle=True, num_workers=1)
    testing_loader = DataLoader(testing_dataset, batch_size=MINIBATCH_SIZE, shuffle=True, num_workers=1)
    temp_mean, temp_std = test_model(model, training_loader)
    mean.append(temp_mean)
    std.append(temp_std)
    temp_mean, temp_std = test_model(model, testing_loader)
    mean.append(temp_mean)
    std.append(temp_std)
    print(f" c - SGD data mean difference = {torch.mean(torch.FloatTensor(mean))}")
    print(f" c - SGD data std = {torch.std(torch.FloatTensor(std))}")
