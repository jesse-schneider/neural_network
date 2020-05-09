import math
import numpy as np
from io import StringIO
import pickle


class NeuralNetwork():
    def __init__(self, num_input, num_hidden, num_output):
        #initalise network
        self.input_to_hiddenW = [[0.1, 0.1],[0.2, 0.1]]
        self.hidden_to_outputW = [[0.1, 0.1], [0.1, 0.2]]
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
        return x * (1 - x)
    
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
                hiddenL[i] = hiddenL[i] + (sample[j] * self.input_to_hiddenW[i][j])
        for i in range(len(hiddenL)):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])
        # print(hiddenL)
        for i in range(len(outputL)):
            for j in range(len(sample)):
                outputL[i] = outputL[i] + (hiddenL[j] * self.hidden_to_outputW[i][j])
        for i in range(len(outputL)):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        # print(outputL)
        for i in range(len(error)):
            error[i] = (0.5 * ((target[i] - outputL[i])**2))
        print("error = ", error[0] + error[1])
        return hiddenL, outputL, error 


    # def backward_pass(self):



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



