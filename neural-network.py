import math
import numpy as np
from io import StringIO
import pickle


class NeuralNetwork():
    def __init__(self, num_input, num_hidden, num_output, input_data, output_data):
        #initalise network
        self.input = input_data
        self.y = output_data
        self.input_to_hiddenW = np.random.random((num_input + 1, 1))
        self.hidden_to_outputW = np.random.random((num_hidden + 1, 1))
        self.output_layer = np.random.random((num_output, 1))
        self.hidden_bias = np.random.random((num_input + 1, 1))
        self.output_bias = np.random.random((num_hidden + 1, 1))
        self.learning_rate = 3
        self.n_epochs = 30
        self.batch_size = 20
        

    #sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    #sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    #training function
    def train(self):
        num_batches = round(self.input.shape[0]/ self.batch_size)

        for i in range(self.epochs):
            for j in range(self.batch_size):

                batch = self.input[(j * self.batch_size):(j+1 * self.batch_size) -1, ]
                current_size = batch.shape[0]


    

    def forward_pass(self):
        self.layer1_beforeB = np.dot(self.input, self.input_to_hiddenW)
        self.layer1 = sigmoid(np.dot(self.layer1_beforeB, self.hidden_bias))

        self.output_beforeB = np.dot(self.layer1, self.hidden_to_outputW)
        self.output = sigmoid(np.dot(self.output_beforeB, self.output_bias))
    
    def backward_pass(self):




inputs = 784
hiddens = 30
outputs = 10

pickledata = open("trainDigitX.pickle", "rb")
pickledata2 = open("trainDigitY.pickle", "rb")
training_data = pickle.load(pickledata)
output_data = pickle.load(pickledata2)

network = NeuralNetwork(inputs, hiddens, outputs, training_data, output_data)


# # Training a neural network network
# n_batches = round(n_samples / batch_size) 
# randomly_initialize_W() 

# for i epoch in range in epochs:
# for i_batch in range batches(n_batches):

# # get all samples in current batch
# current_batch = batches[i_batch]

# #number of samples the current batch 
# current_batch_size = current_batch.shape[0]

# sum_gradw = zero size(W})

# # Loop though all samples in current batch 
# for (x, y) for (x, y) in current batch:
#     cost, activations = forward_pass(x, y) 
#     gradw_x = backward pass(x, y)

#     sum_gradw = sum_gradw + gradw_x

# # average of gradients over the current batch grade 
# gradW_batch = batch sum_gradw / current batch_size

# # update the parameters 
# W = W - learning rate * gradw_batch






# def forward():

# def backward():



