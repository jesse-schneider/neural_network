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
        self.n_epochs = 30
        self.batch_size = 20
        

    #sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    #sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
    
    # #training function
    # def train(self):
    #     num_batches = round(self.input.shape[0]/ self.batch_size)

    #     for i in range(self.epochs):
    #         for j in range(self.batch_size):

    #             batch = self.input[(j * self.batch_size):(j+1 * self.batch_size) - 1, ]
    #             current_size = batch.shape[0]

    #             sum_gradients = np.zeros((self.hi))



    def forward_pass(self, sample, target):
        outputL = np.zeros(2)
        hiddenL = np.zeros(2)
        error = np.zeros(2)
        for i in range(len(hiddenL)):
            for j in range(len(sample)):
                hiddenL[i] = hiddenL[i] + (sample[j] * self.hiddenWeights[i][j])
        for i in range(len(hiddenL)):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])
        # print(hiddenL)

        for i in range(len(outputL)):
            for j in range(len(sample)):
                outputL[i] = outputL[i] + (hiddenL[j] * self.outputWeights[i][j])
        for i in range(len(outputL)):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        # print(outputL)

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
                error = -(target[j] - out_o[j]) * (out_o[j] * (1 - out_o[j])) * self.outputWeights[i][j]
                errorSum += error
                if j == outputs - 1:
                    error_outh.append(errorSum)
        print(error_outh)

        # e1_h1 = -(target[0] - out_o[0]) * (out_o[0] * (1 - out_o[0])) * self.outputWeights[0][0] #weight 5
        # e2_h1 = -(target[1] - out_o[1]) * (out_o[1] * (1 - out_o[1])) * self.outputWeights[0][1] #weight 5

        # e1_h2 = -(target[0] - out_o[0]) * (out_o[0] * (1 - out_o[0])) * self.outputWeights[1][0] #weight 7
        # e2_h2 = -(target[1] - out_o[1]) * (out_o[1] * (1 - out_o[1])) * self.outputWeights[1][1] #weight 8
        # e_outh1 = e1_h1 + e2_h1
        # e_outh2 = e1_h2 + e2_h2
        # error_outh = [e_outh1, e_outh2]

        for i in range(len(inputs)):
            for j in range(len(self.hiddenWeights[i])):
                weight = error_outh[j] * (out_h[j] * (1 - out_h[j])) * inputs[i]
                print("weights 1 - 4: ", format(weight, '.8f'))
                weights.append(weight)
        
        for i in range(len(inputs)):
            for j in range(len(self.outputWeights[i])):
                weight = -(target[j] - out_o[j]) * (out_o[j] * (1 - out_o[j])) * (out_h[i])
                print("weights 5 - 8: ", format(weight, '.8f'))
                weights.append(weight)


        for i in range(len(self.hidden_bias)):
                weight = (out_h[i] * (1 - out_h[i])) * error_outh[i]
                print("weights 9 - 10: ", format(weight, '.8f'))
                weights.append(weight)

        for i in range(len(self.output_bias)):
                weight = (out_o[i] * (1 - out_o[i])) * error_outh[i]
                print("weights 11 - 12: ", format(weight, '.8f'))
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
for i in range(inputs):
    out_h, out_o, error = network.forward_pass(training_data[i], output_data[i])
    gradients = network.backward_pass(out_h, out_o, output_data[i], training_data[i])



