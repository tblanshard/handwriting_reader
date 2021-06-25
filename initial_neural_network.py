import numpy
import scipy.special


class neuralNetwork:

		def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
				# store the inputs that define the size and structure of the neural network

				self.inodes = input_nodes
				self.hnodes = hidden_nodes
				self.onodes = output_nodes

				self.lr = learning_rate

				# the weights of the links
				# weights inside the arrays are w_i_j where link is from node to node j in the next layer

				#self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
				#self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

				self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
				self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

				# activation function - sigmoid function
				# using lambda function - takes x and returns scipty.special.expit(x)

				self.activation_function = lambda x: scipy.special.expit(x)

		def train(self, inputs_list, target_list):

				# convert the inputs list to a 2d array

				inputs = numpy.array(inputs_list, ndmin=2).T
				targets = numpy.array(target_list, ndmin=2).T

				# calculate the signals into the hidden layer

				hidden_inputs = numpy.dot(self.wih, inputs)
				hidden_outputs = self.activation_function(hidden_inputs)

				# calculate the signals into the final output layer

				final_inputs = numpy.dot(self.who, hidden_outputs)
				final_outputs = self.activation_function(final_inputs)

				# need to calculate the error

				output_errors = targets - final_outputs

				# hidden layer error
				# output_errors split by weights and recombined at the hidden nodes

				hidden_errors = numpy.dot(self.who.T, output_errors)

				#update the weights for the links between the hidden and the output layers

				self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))

				self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))



		def query(self, inputs_list):

				# convert the inputs list to a 2d array

				inputs = numpy.array(inputs_list, ndmin=2).T

				# calculate signals into hidden layer

				hidden_inputs = numpy.dot(self.wih, inputs)
				hidden_outputs = self.activation_function(hidden_inputs)

				# calculate signals into final layer

				final_inputs = numpy.dot(self.who, hidden_outputs)
				final_outputs = self.activation_function(final_inputs)

				return final_inputs



def main():
		# define the size of our neural network and initialise it

		input_nodes = 3
		hidden_nodes = 3
		output_nodes = 3

		learning_rate = 0.3

		n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

		print(n.query([1.0, 0.5, -1.5]))


main()
