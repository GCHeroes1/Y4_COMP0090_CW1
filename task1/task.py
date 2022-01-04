# https://pytorch.org/vision/stable/datasets.html
# https://pytorch.org/docs/stable/data.html
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
import torch.linalg
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

dtype = torch.float32


def polynomial_fun(weight_vector, scalar_variable_x):
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
	def __init__(self, degree_=2):
		super(PolyModel, self).__init__()
		self.poly = nn.Linear(degree_, 1)

	def forward(self, x):
		out = self.poly(x)
		return out


def make_features(x, degree):
	'''Builds features i.. a matrix with columns [x,x^2,x^3].'''
	# x = x.unsqueeze(1)
	return torch.cat([x ** i for i in range(1, degree + 1)], 1)


def create_features(X, degree=2):
	"""Creates the polynomial features

    Args:
        X: A torch tensor for the data.
        degree: A integer for the degree of the generated polynomial function.
    """
	if len(X.shape) == 1:
		X = X.unsqueeze(1)
	# Concatenate a column of ones to has the bias in X
	ones_col = torch.ones((X.shape[0], 1), dtype=torch.float32)
	X_d = torch.cat([ones_col, X], axis=1)
	for i in range(1, degree):
		X_pow = X.pow(i + 1)
		X_d = torch.cat([X_d, X_pow], axis=1)
	return X_d


def fit_polynomial_sgd(x_coordinates, t_target_values, degree, learning_rate, minibatch_size):
	# degree = 3
	model = nn.Sequential(
			nn.Linear(degree, 1)
	)
	# model = PolyModel(degree_=degree)

	features = make_features(x_coordinates, degree=degree)
	dataset = TensorDataset(features, t_target_values)

	training_loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True, num_workers=1)

	# loss and optimiser
	print(learning_rate)
	criterion = torch.nn.MSELoss(reduction='mean')
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

	epochs = 500
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
			# if i % 10 == 9:  # print every 5 mini-batches
			#     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
			#     running_loss = 0.0
		loss = float(running_loss) / float(total)
		if epoch % 50 == 49:
			print(f"loss is {str(loss)} after {str(epoch + 1)} epochs")
	print('Training done.')

	for i, data in enumerate(training_loader, 0):
		inputs, labels = data
		pred = model(inputs)
		print("Prediction is :\n", pred)
		print("Actual targets is :\n", labels)
		break

	# save trained model
	torch.save(model.state_dict(), './task1/SGD_model.pt')
	print('Model saved.')
	return model


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
	MINIBATCH_SIZE = 4

	weight_vector = torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1)  # Add Second Dimension
	# print(weight_vector)

	x_training_coordinates = torch.randint(low=-20, high=20, size=(100,), dtype=dtype).unsqueeze(1)  # ([100, 1])
	x_testing_coordinates = torch.randint(low=-20, high=20, size=(50,), dtype=dtype).unsqueeze(1)  # ([50, 1])
	# print(x_training_coordinates.shape, x_testing_coordinates.shape)  # ([100, 1]) ([50, 1])

	# feature map of x_training_coordinates and x_testing_coordinates
	x_training_map = make_features(x_training_coordinates, DEGREE)
	x_testing_map = make_features(x_testing_coordinates, DEGREE)
	# print(x_training_map.shape, x_testing_map.shape)  # ([100, 4]) ([50, 4])

	t_training_coordinates = polynomial_fun(weight_vector, x_training_map)
	noise = torch.normal(mean=0.0, std=0.2, size=t_training_coordinates.shape, dtype=dtype)
	t_training_coordinates += noise
	# print(t_training_coordinates.shape)  # ([100, 1])

	t_testing_coordinates = polynomial_fun(weight_vector, x_testing_map)
	noise = torch.normal(mean=0.0, std=0.2, size=t_testing_coordinates.shape, dtype=dtype)
	t_testing_coordinates += noise
	# print(t_testing_coordinates.shape)  # ([50, 1])
	# print("done sampling")

	# # weight_vector = torch.reshape(torch.arange(1, 5), (4, 1))  # tensor([[1], [2], [3], [4]])
	# # weight_vector = torch.transpose(torch.arange(1, 5), -1, 0)  # tensor([1, 2, 3, 4])
	# # print(weight_vector)
	# # print(torch.reshape(torch.arange(1, 5), (4, 1)))
	#
	# weight_vector = torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1)  # Add Second Dimension
	#
	# # X = make_features(torch.Tensor(5), 4)
	#
	# # print(f(weight_vector, X))
	#
	# t_training_coordinates = torch.tensor([])
	# for coordinate_index, x_value in enumerate(x_training_coordinates):
	#     noise = torch.normal(mean=0.0, std=0.2, size=(4,))
	#     y = noise + polynomial_fun(weight_vector, x_value)
	#     t_training_coordinates = torch.cat((t_training_coordinates, torch.Tensor(y)))
	#
	# t_testing_coordinates = torch.tensor([])
	# for coordinate_index, x_value in enumerate(x_testing_coordinates):
	#     noise = torch.normal(mean=0.0, std=0.2, size=(1,))
	#     y = noise + polynomial_fun(weight_vector, x_value)
	#     t_testing_coordinates = torch.cat((t_testing_coordinates, torch.Tensor(y)))

	LS_optimum_weight_vector = fit_polynomial_ls(x_training_coordinates, t_training_coordinates, DEGREE)
	print(f"the LS optimised weights are {str(LS_optimum_weight_vector)}")

	# y_hat_training = torch.tensor([])
	# for coordinate_index, x_value in enumerate(x_training_coordinates):
	#     y_hat = polynomial_fun(LS_optimum_weight_vector, x_value)
	#     y_hat_training = torch.cat((y_hat_training, y_hat.reshape(1)))
	# y_hat_testing = torch.tensor([])
	# for coordinate_index, x_value in enumerate(x_testing_coordinates):
	#     y_hat = polynomial_fun(LS_optimum_weight_vector, x_value)
	#     y_hat_testing = torch.cat((y_hat_testing, y_hat.reshape(1)))

	SGD_optimum_weight_vector = fit_polynomial_sgd(x_training_coordinates, t_training_coordinates, DEGREE,
	                                               LEARNING_RATE, MINIBATCH_SIZE)
	print(SGD_optimum_weight_vector)
