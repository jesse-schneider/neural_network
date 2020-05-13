import math
import numpy as np
from io import StringIO
import pickle


class NeuralNetwork():
    def __init__(self, num_input, num_hidden, num_output):
        #initalise network
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.hiddenWeights = [[0.1, 0.1],[0.2, 0.1]]
        self.outputWeights = [[0.1, 0.1], [0.1, 0.2]]
        self.hidden_bias = [0.1, 0.1]
        self.output_bias = [0.1, 0.1]
        self.learning_rate = 3
        self.n_epochs = 5
        self.batch_size = 2
        

    #sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    #sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def cost_function(self, x, y):
       return (1/2*self.num_input) * ((x - y)**2)


    def forward_pass(self, sample, target):
        outputL = np.zeros(self.num_output)
        hiddenL = np.zeros(self.num_hidden)
        error = np.zeros(self.num_output)

        #compute all inputs x weights into each hidden node
        for i in range(self.num_input):
            for j in range(self.num_hidden):
                hiddenL[j] = hiddenL[j] + (sample[i] * self.hiddenWeights[i][j])
        for i in range(self.num_hidden):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])

        #compute all hidden x weights into each output node
        for i in range(self.num_hidden):
            for j in range(self.num_output):
                outputL[j] = outputL[j] + (hiddenL[i] * self.outputWeights[i][j])
        for i in range(self.num_output):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        # print(outputL)

        #calculate each output node error
        for i in range(len(error)):
            error[i] = self.cost_function(target[i], outputL[i])
        return hiddenL, outputL, error


    def backward_pass(self, out_h, out_o, target, inputs):
        weights = []
        error_outh = []

        for i in range(self.num_hidden):
            errorSum = 0
            for j in range(self.num_output):
                error = -(target[j] - out_o[j]) * self.sigmoid_derivative(out_o[j]) * self.outputWeights[i][j]
                errorSum += error
                if j == outputs - 1:
                    error_outh.append(errorSum)

        for i in range(self.num_input):
            for j in range(len(self.hiddenWeights[i])):
                weight = error_outh[j] * self.sigmoid_derivative(out_h[j]) * inputs[i]
                weights.append(weight)
        
        for i in range(self.num_hidden):
            for j in range(len(self.outputWeights[i])):
                weight = -(target[j] - out_o[j]) * self.sigmoid_derivative(out_o[j]) * (out_h[i])
                weights.append(weight)


        for i in range(len(self.hidden_bias)):
                weight = self.sigmoid_derivative(out_h[i]) * error_outh[i]
                weights.append(weight)

        for i in range(len(self.output_bias)):
                weight = self.sigmoid_derivative(out_o[i]) * error_outh[i]
                weights.append(weight)
        return weights

training_data = [[0.1, 0.1], [0.1, 0.2]]
output_data = [[1, 0], [0, 1]]

inputs = 2
hiddens = 2
outputs = 2

# pickledata = open("trainDigitX.pickle", "rb")
# pickledata2 = open("trainDigitY.pickle", "rb")
# training_data = pickle.load(pickledata)
# output_data = pickle.load(pickledata2)

network = NeuralNetwork(inputs, hiddens, outputs)
num_batchs = len(training_data) // network.batch_size

for epoch in range(network.n_epochs):
    for batch in range(num_batchs):
        print("batch number: ", batch)

        current = training_data[(batch * network.batch_size): (batch + 1 * network.batch_size)]

        current_size = len(current)

        #initialise gradients sum and average arrays
        gradients_sum = [0 for i in range((inputs * hiddens) + (hiddens * outputs) + hiddens + outputs)]
        average_gradients = [0 for i in range((inputs * hiddens) + (hiddens * outputs) + hiddens + outputs)]

        for i in range(len(current)):
            out_h, out_o, error = network.forward_pass(training_data[i], output_data[i])
            gradients = network.backward_pass(out_h, out_o, output_data[i], training_data[i])

            #add gradients from back pass to sum total
            for i in range(len(gradients)):
                gradients_sum[i] += gradients[i]
        
        #get the average gradients from the batch
        for i in range(len(average_gradients)):
            average_gradients[i] = gradients_sum[i] / current_size

        for i in range(network.num_hidden):
            for j in range(len(network.hiddenWeights)):
                network.hiddenWeights[j][i] -= (network.learning_rate * average_gradients[j])

        ind = network.num_hidden
        for i in range(network.num_output):
            for j in range(len(network.outputWeights)):
                network.outputWeights[j][i] -= (network.learning_rate * average_gradients[ind+j])
 
        ind = network.num_hidden + network.num_output
        for i in range(len(network.hidden_bias)):
            network.hidden_bias[i] -= (network.learning_rate * average_gradients[ind+i])

        ind = (2 * network.num_hidden) + network.num_output
        for i in range(len(network.output_bias)):
            network.output_bias[i] -= (network.learning_rate * average_gradients[ind+i])

        errorSum = 0
        for i in range(len(error)):
            errorSum += error[i]
        av_error = errorSum / len(error)
        
        print("Average Error: ", av_error)

        # print("hidden weights: ", network.hiddenWeights)
        # print("output weights: ", network.outputWeights)
        # print("hidden bias: ", network.hidden_bias)
        # print("output bias: ", network.output_bias)
