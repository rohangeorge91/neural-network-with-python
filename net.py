# Import numpy
import numpy as np

# activation function (Sigmoid)
def nonlin(x, deriv=False):
	if(deriv == True):
		return x * (1-x)
	return 1/(1+np.exp(-x))

# Input
X = np.array([
	[0,0,1],
	[0,1,1],
	[1,0,1],
	[1,1,1]
	])
# Actual output
Y = np.array([[0],[1],[1],[0]])

# fixing the seed values in order to get the same random numbers
np.random.seed(1)

# synapses
syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1

# iterations
for j in range(10000):
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))

	# difference between predicted output and actual output
	l2_err = Y - l2

	# printing error for every 100 iterations
	if(j % 100) == 0:
		print("Error : ",np.mean(l2_err))

	# using rate of change of error to adjust weights
	l2_delta = l2_err * nonlin(l2,True)
	l1_err = np.dot(l2_delta,syn1.T)
	l1_delta = l1_err * nonlin(l1,True)

	# updating synapses
	syn0 += np.dot(l0.T,l1_delta)
	syn1 += np.dot(l1.T,l2_delta)

# printing final predicted output which will be closer to actual output
print(l2)