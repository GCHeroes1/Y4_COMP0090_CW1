import torch
import torch.linalg
from torch.utils.data import DataLoader, TensorDataset

import os
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

dtype = torch.float32

def polynomial_fun(weight_vector, scalar_variable_x):
	y = 0
	# M = len(weight_vector)-1
	for m_index, m_values in enumerate(weight_vector):
		# print(m_index)
		y += m_values * (scalar_variable_x ** m_index - 1)
	return y

def fit_polynomial_ls(x_coordinates, t_target_values, ls_degree):  ## the weights are output in backwards order, the lowest degree is the start of the vector
	ones = torch.ones_like(x_coordinates)
	x_degrees_to_be_added = list()
	x_degrees_to_be_added.append(ones)
	x_degrees_to_be_added.append((x_coordinates))
	x = torch.stack((ones, x_coordinates), 1)
	for i in range(1, ls_degree+1):
		if i > 1:
			x_degrees_to_be_added.append(x_coordinates ** i)
	x_degrees = torch.stack(x_degrees_to_be_added, 1)
	# print(x_degrees)
	# weights = torch.lstsq(t_target_values, x_degrees).solution[:x_degrees.size(1)]

	x_degrees_transpose = torch.transpose(x_degrees, 0, 1)
	comp1 = torch.matmul(x_degrees_transpose, x_degrees)
	comp2 = torch.inverse(comp1)
	comp3 = torch.matmul(x_degrees_transpose, t_target_values)
	weights = torch.matmul(comp2, comp3)

	return weights

def mean_loss(y, y_hat):
	return torch.sum((y - y_hat)**2)/y.size(dim=0)

def model(x_value, weights):
	y = 0
	for weight_index, weight in enumerate(weights):
		# y = b + x + x^2 + x^3 ...
		y += weight * (x_value ** weight_index - 1)
	return y

def fit_polynomial_sgd(x_coordinates, t_target_values, degree, learning_rate, minibatch_size):
	epochs = 10

	model = torch.nn.Sequential(
			torch.nn.Linear(degree, 1)
	)

	dataset = TensorDataset(x_coordinates, t_target_values)
	training_loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True, num_workers=2)

	# loss and optimiser
	criterion = torch.nn.MSELoss(reduction='mean')
	optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

	for epoch in range(epochs):
		# print(f'Starting epoch {epoch + 1}')
		running_loss = 0.0

		for i, data in enumerate(training_loader, 0):
			inputs, labels = data
			# print(f"inputs are {str(inputs)}")
			# print(f"labels are {str(labels)}")
			# print(f"before unsqueeze {str(labels)}")
			# labels = labels.unsqueeze(-1)
			# print(f"after unsqueeze {str(labels)}")
			optimiser.zero_grad()

			# print(f"inputs are {str(inputs)}")
			# for param in model.parameters():
			# 	print(param.data)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimiser.step()

			running_loss += loss.item()

			minibatch = 5
			if i % minibatch == minibatch-1:  # print every 5 mini-batches
				# print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / minibatch))
				running_loss = 0.0
	print('Training done.')

	# for x, y in training_loader:
	# 	pred = model(inputs)
	# 	print("Prediction is :\n", pred)
	# 	print("Actual targets is :\n", y)
	# 	break
	for param in model.parameters():
		print(param.data)
	return inputs

def fit_polynomial_sgd_2(x_coordinates, t_target_values, degree, learning_rate, minibatch_size):
	dataset = TensorDataset(x_coordinates, t_target_values)
	epochs = 10
	training_loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

	# weights_ = torch.randint(low=1, high=4, size=(degree+1,), dtype=dtype, requires_grad=True)
	weights_ = torch.arange(1, degree+2, requires_grad=True, dtype=dtype)

	# weights = torch.rand(degree+1, requires_grad=True, dtype=dtype)
	# print("initial weights of SGD are " + str(weights_))

	for _ in range(epochs):
		for x, y in training_loader:
			prediction = polynomial_fun(weights_, x) # consider changing to polynomial _ fun
			loss = mean_loss(y, prediction)
			loss.backward()
			with torch.no_grad():
				weights_.grad.zero_()

	# for x, y in training_loader:
	# 	pred = model(x, weights)
	# 	print("Prediction is :\n", pred)
	# 	print("Actual targets is :\n", y)
	# 	break

	return weights_

if __name__ == '__main__':
	dtype = torch.float32
	data_size = 10
	x_coordinates = torch.randint(low=0, high=10, size=(10,), dtype=dtype)
	t_coordinates = torch.randint(low=0, high=10, size=(10,), dtype=dtype)
	# degree = 2

	# print("resulting weight of LS fit: " + str(fit_polynomial_ls(x_coordinates, t_coordinates, degree)))

	# print("resulting weight of SGD fit: " + str(fit_polynomial_sgd(x_coordinates, t_coordinates, degree, learning_rate, minibatch_size)))
	# data = torch.stack((x_coordinates, t_coordinates), 1)

	x_training_coordinates = torch.randint(low=-20, high=20, size=(100,), dtype=dtype)
	x_testing_coordinates = torch.randint(low=-20, high=20, size=(50,), dtype=dtype)

	weight_vector = torch.transpose(torch.arange(1, 5), -1, 0)
	# print(w)
	# weight = torch.transpose(torch.Tensor([1, 2, 3, 4]))
	# print(weight)

	t_training_coordinates = torch.tensor([])
	for coordinate_index, x_value in enumerate(x_training_coordinates):
		noise = torch.normal(mean=0.0, std=0.2, size=(1,))
		y = noise + polynomial_fun(weight_vector, x_value)
		t_training_coordinates = torch.cat((t_training_coordinates, torch.Tensor(y)))

	t_testing_coordinates = torch.tensor([])
	for coordinate_index, x_value in enumerate(x_testing_coordinates):
		noise = torch.normal(mean=0.0, std=0.2, size=(1,))
		y = noise + polynomial_fun(weight_vector, x_value)
		t_testing_coordinates = torch.cat((t_testing_coordinates, torch.Tensor(y)))

	degree = 5
	learning_rate = 0.001
	minibatch_size = 5

	LS_optimum_weight_vector = fit_polynomial_ls(x_training_coordinates, t_training_coordinates, degree)
	print(f"the LS optimised weights are {str(LS_optimum_weight_vector)}")

	y_hat_training = torch.tensor([])
	for coordinate_index, x_value in enumerate(x_training_coordinates):
		# print(x_training_coordinates)
		y_hat = polynomial_fun(LS_optimum_weight_vector, x_value)
		# print(y_hat_training)
		# print(y_hat)
		y_hat_training = torch.cat((y_hat_training, y_hat.reshape(1)))
	y_hat_testing = torch.tensor([])
	for coordinate_index, x_value in enumerate(x_testing_coordinates):
		y_hat = polynomial_fun(LS_optimum_weight_vector, x_value)
		y_hat_testing = torch.cat((y_hat_testing, y_hat.reshape(1)))

	SGD_optimum_weight_vector = fit_polynomial_sgd(x_training_coordinates, t_training_coordinates, degree, learning_rate, minibatch_size)
	print(SGD_optimum_weight_vector)