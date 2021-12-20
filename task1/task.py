import torch
import torch.linalg
from torch.utils.data import DataLoader, TensorDataset

dtype = torch.float32

def polynomial_fun(weight_vector, scalar_variable_x):
	y = 0
	M = len(weight_vector)-1
	# print(f"M is {str(M)}")
	for m_index, m_values in enumerate(weight_vector):
		# print(m_index)
		y += weight_vector[m_index] * scalar_variable_x ** m_values
	return y

def fit_polynomial_ls(x_coordinates, t_target_values, degree):  ## the weights are output in backwards order, the lowest degree is the start of the vector
	ones = torch.ones_like(x_coordinates)
	x_degrees_to_be_added = list()
	x_degrees_to_be_added.append(ones)
	x_degrees_to_be_added.append((x_coordinates))
	x = torch.stack((ones, x_coordinates), 1)
	for i in range(1, degree+1):
		if i > 1:
			x_degrees_to_be_added.append(x_coordinates ** i)
	x_degrees = torch.stack(x_degrees_to_be_added, 1)
	weights = torch.lstsq(t_coordinates, x_degrees).solution[:x_degrees.size(1)]

	# x_degrees_transpose = torch.transpose(x_degrees, 0, 1)
	# comp1 = torch.matmul(x_degrees_transpose, x_degrees)
	# comp2 = torch.inverse(comp1)
	# comp3 = torch.matmul(x_degrees_transpose, t_coordinates)
	# weights = torch.matmul(comp2, comp3)

	return weights

def mean_loss(y, y_hat):
	return torch.sum((y - y_hat)**2)/y.size(dim=0)

def model(x_value, weights):
	# model = torch.nn.Sequential(
	# 		torch.nn.Linear(degree, 1)
	# )
	y = 0
	for weight_index, weight in enumerate(weights):
		# y = b + x + x^2 + x^3 ...
		y += weight * (x_value ** weight_index - 1)
	return y

def fit_polynomial_sgd(x_coordinates, t_target_values, degree, learning_rate, minibatch_size):
	# dataset = torch.stack((x_coordinates, t_coordinates), 1)
	# print(dataset)
	dataset = TensorDataset(x_coordinates, t_target_values)
	print(dataset)
	epochs = 3
	# batch_size = degree
	training_loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

	# weights = torch.randint(low=0, high=5, size=(degree+1,), dtype=dtype, requires_grad=True)
	weights = torch.rand(degree+1, requires_grad=True)
	print(weights)

	for _ in range(epochs):
		for x, y in training_loader:
			prediction = model(x, weights)
			loss = mean_loss(y, prediction)
			loss.backward()
			with torch.no_grad():
				weights -= (weights.grad * -learning_rate)
				weights.grad.zero_()
		print(f"Epoch {_}/{epochs}: Loss: {loss}")

	for x, y in training_loader:
		pred = model(x, weights)
		print("Prediction is :\n", pred)
		print("\nActual targets is :\n", y)
		break

	return weights


if __name__ == '__main__':
	dtype = torch.float32
	noise = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor(3.20))
	noise = noise.sample()

	x_coordinates = torch.randint(low=0, high=10, size=(10,), dtype=dtype)
	t_coordinates = torch.randint(low=0, high=10, size=(10,), dtype=dtype)
	degree = 2

	# print(fit_polynomial_ls(x_coordinates, t_coordinates, degree))

	learning_rate = 0.1
	minibatch_size = 2

	print(fit_polynomial_sgd(x_coordinates, t_coordinates, degree, learning_rate, minibatch_size))
	# data = torch.stack((x_coordinates, t_coordinates), 1)



