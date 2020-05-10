import math
import numpy as np
from io import StringIO
import pickle


class NeuralNetwork():
    def __init__(self, num_input, num_hidden, num_output):
        #initalise network
        self.hiddenWeights = [[0.1, 0.1],[0.2, 0.1]]
        self.outputWeights = [[0.1, 0.1], [0.1, 0.2]]
        self.hidden_bias = [0.1, 0.1]
        self.output_bias = [0.1, 0.1]
        self.learning_rate = 3
        self.n_epochs = 100
        self.batch_size = 2
        

    #sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    #sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def forward_pass(self, sample, target):
        outputL = np.zeros(2)
        hiddenL = np.zeros(2)
        error = np.zeros(2)
        for i in range(len(hiddenL)):
            for j in range(len(sample)):
                hiddenL[i] = hiddenL[i] + (sample[j] * self.hiddenWeights[i][j])
        for i in range(len(hiddenL)):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])
        # print("out_h: ", hiddenL)

        for i in range(len(outputL)):
            for j in range(len(sample)):
                outputL[i] = outputL[i] + (hiddenL[j] * self.outputWeights[i][j])
        for i in range(len(outputL)):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        # print("out_o: ", outputL)

        for i in range(len(error)):
            error[i] = (0.5 * ((target[i] - outputL[i])**2))
        # print("error = ", error[0] + error[1])
        return hiddenL, outputL, error 


    def backward_pass(self, out_h, out_o, target, inputs):
        weights = []
        error_outh = []

        for i in range(hiddens):
            errorSum = 0
            for j in range(outputs):
                error = -(target[j] - out_o[j]) * self.sigmoid_derivative(out_o[j]) * self.outputWeights[i][j]
                errorSum += error
                if j == outputs - 1:
                    error_outh.append(errorSum)

        for i in range(len(inputs)):
            for j in range(len(self.hiddenWeights[i])):
                weight = error_outh[j] * self.sigmoid_derivative(out_h[j]) * inputs[i]
                weights.append(weight)
        
        for i in range(len(inputs)):
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

        k = 0
        for i in range(len(network.hiddenWeights)):
            for j in range(len(network.hiddenWeights[i])):
                if j % 2 == 1:
                    k += 1
                network.hiddenWeights[i][j] -= (network.learning_rate * average_gradients[k+j])
                if j % 2 == 1:
                    k -= 1
            k += hiddens

        for i in range(len(network.outputWeights)):
            for j in range(len(network.outputWeights[i])):
                if j % 2 == 1:
                    k += 1
                network.outputWeights[i][j] -= (network.learning_rate * average_gradients[k+j])
                if j % 2 == 1:
                    k -= 1
            k += outputs

        for i in range(len(network.hidden_bias)):
            network.hidden_bias[i] -= (network.learning_rate * average_gradients[k+i])
        k += hiddens

        for i in range(len(network.output_bias)):
            network.output_bias[i] -= (network.learning_rate * average_gradients[k+i])

        errorSum = 0
        for i in range(len(error)):
            errorSum += error[i]
        av_error = errorSum / len(error)
        
        print("Average Error: ", av_error)

        # print("hidden weights: ", network.hiddenWeights)
        # print("output weights: ", network.outputWeights)
        # print("hidden bias: ", network.hidden_bias)
        # print("output bias: ", network.output_bias)
