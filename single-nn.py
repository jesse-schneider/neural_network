from numpy import exp, array, random, dot


#this is a single neuron neural network

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # we assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

        # The sigmoid function, which describes an S shaped curve
        # we pass the weighted sum of the inputs through this function
        # to normalise them between 0 and 1
    def __sigmoid(self, x):
        return 1/(1 + exp(-x))

    # gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
            for iteration in range(number_of_training_iterations):
                # pass the training set through our NN
                output = self.predict(training_set_inputs)

                # calculate the error
                error = training_set_outputs - output

                # multiply the error by the input and again by the gradient of the sigmoid curve
                adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

                # adjust the weights
                self.synaptic_weights += adjustment



    def predict(self, inputs):
        # pass inputs through our NN (the single nueron)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))



# initalise a single neuron NN
neural_network = NeuralNetwork()

print('Random starting synaptic weights: ')
print(neural_network.synaptic_weights)

# the training set. We have 4 examples, each consisting of 3 input values
# and 1 output value
training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_set_outputs = array([[0,1,1,0]]).T

# traing the NN using training set
# Do this 10,000 times and make small adjustments each time
neural_network.train(training_set_inputs, training_set_outputs, 10000)

print('New synaptic weights after training: ')
print(neural_network.synaptic_weights)

# Test the NN
print('predicting ')
print(neural_network.predict(array([1,0,0])))




# def initialise_network(num_inputs, num_hidden, num_outputs):
#     network = list()
#     hidden_layer = [{'weights':random() for i in range(num_inputs + 1)} for i in range(num_hidden)]
#     network.append(hidden_layer)
#     output_layer = [{'weights':random() for i in range(num_hidden + 1)} for i in range(num_outputs)]
#     network.append(output_layer)
#     return network
