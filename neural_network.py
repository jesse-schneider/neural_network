import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

"""
    to run:
    python3 neural_network.py 784 30 10 TrainDigitX.csv TrainDigitY.csv TestDigitX.csv TestDigitY.csv TestDigitX2.csv
"""

class NeuralNetwork():
    def __init__(self, num_input, num_hidden, num_output):
        #initalise network
        self.num_input = int(num_input)
        self.num_hidden = int(num_hidden)
        self.num_output = int(num_output)
        self.hidden_weights = [[np.random.randn() for i in range(self.num_hidden)] for j in range(self.num_input)]
        self.output_weights = [[np.random.randn() for i in range(self.num_output)] for j in range(self.num_hidden)]
        self.hidden_bias = [np.random.randn() for i in range(self.num_hidden)]
        self.output_bias = [np.random.randn() for i in range(self.num_output)]
        self.learning_rate = 3
        self.n_epochs = 5
        self.batch_size = 20
        

    #sigmoid activation function
    def sigmoid(self, x):
        return (1.0 / (1.0 + math.exp(-x)))

    #sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def cost_function(self, x):
        return (1/self.num_input) * (x**2)

    def test_data(self, test_data, test_data_out):
        #run test data set, test predictions
        predictions = []
        for i in range(len(test_data)):
            predict = network.predict(test_data[i])
            max = 0
            for j in range(len(predict)):
                if predict[j] > predict[max]:
                    max = j
            print("prediction", i, ":", max, "    actual value:", test_data_out[i])
            predictions.append(max)
        np.savetxt('PredictDigitY.csv.gz', predictions, fmt='%0f')
        return predictions
    
    def predict_data(self, test_data):
        #run test data set, predict unknown answers
        predictions = []
        for i in range(len(test_data)):
            predict = network.predict(test_data[i])
            max = 0
            for j in range(len(predict)):
                if predict[j] > predict[max]:
                    max = j
            print("prediction", i, ":", max)
            predictions.append(max)
        return predictions


    def update_weights(self, output_gradients, hidden_gradients, hidden_bias_grad, output_bias_grad, batch_size):
        
        #update hidden -> output weights
        for i in range(self.num_hidden):
            for j in range(self.num_output):
                self.output_weights[i][j] -= (self.learning_rate * (output_gradients[i][j] / batch_size))

        #update input -> hidden weights
        for i in range(self.num_input):
            for j in range(self.num_hidden):
                self.hidden_weights[i][j] -= (self.learning_rate * (hidden_gradients[i][j] / batch_size))
 
        #update hidden bias
        for i in range(len(self.hidden_bias)):
            self.hidden_bias[i] -= (self.learning_rate * (hidden_bias_grad[i] / batch_size))

        #update output bias
        for i in range(len(network.output_bias)):
            self.output_bias[i] -= (self.learning_rate * (output_bias_grad[i] /batch_size))

    def predict(self, sample):
        #predict function runs a forward pass without calculating error
        outputL = [0 for i in range(self.num_output)]
        hiddenL = [0 for i in range(self.num_hidden)]

        for i in range(self.num_input):
            for j in range(self.num_hidden):
                hiddenL[j] = hiddenL[j] + (sample[i] * self.hidden_weights[i][j])
        for i in range(self.num_hidden):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])

        for i in range(self.num_hidden):
            for j in range(self.num_output):
                outputL[j] = outputL[j] + (hiddenL[i] * self.output_weights[i][j])
        for i in range(self.num_output):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        return outputL


    def forward_pass(self, sample, target):
        #lists to store the results at each stage of the forward pass
        outputL = [0 for i in range(self.num_output)]
        hiddenL = [0 for i in range(self.num_hidden)]
        error = [0 for i in range(self.num_output)]

        #pass input layer into the hidden layer
        for i in range(self.num_input):
            for j in range(self.num_hidden):
                hiddenL[j] += sample[i] * self.hidden_weights[i][j]

        #add bias and then perform sigmoid activation on hidden layer
        for i in range(self.num_hidden):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])

        #pass hidden layer into output layer
        for i in range(self.num_hidden):
            for j in range(self.num_output):
                outputL[j] = outputL[j] + (hiddenL[i] * self.output_weights[i][j])

        #add bias and then perform sigmoid activation on output layer
        for i in range(self.num_output):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])

        #calculate error in output node by finding target - output
        for i in range(len(error)):
            error[i] = (target[i] - outputL[i])
        return hiddenL, outputL, error


    def backward_pass(self, out_h, out_o, error, inputs, target):
        #output arrays size of weights to store their gradients
        hidden_output_grad = [[0 for i in range(self.num_output)] for j in range(self.num_hidden)]
        input_hidden_grad = [[0 for i in range(self.num_hidden)] for j in range(self.num_input)]
        hidden_bias_grad = [0 for i in range(self.num_hidden)]
        output_bias_grad = [0 for i in range(self.num_output)]

        #error arrays used to store different parts of the calculations 
        out_error_derivative = [0 for i in range(self.num_output)]
        hidden_error = [[0 for i in range(self.num_output)] for j in range(self.num_hidden)]
        hidden_error_summed = [0 for i in range(self.num_hidden)]
        input_error = [0 for i in range(self.num_hidden)]
        

        #calculate error from output node
        for i in range(len(error)):
            out_error_derivative[i] = error[i] * self.sigmoid_derivative(out_o[i])

        #finding error in weights from hidden -> output
        for i in range(self.num_hidden):
            for j in range(len(out_error_derivative)):
                hidden_output_grad[i][j] = out_error_derivative[j] * out_h[i] * (-1)

        #find error caused by each weight in hidden layer
        for i in range(len(self.output_weights)):
            for j in range(len(self.output_weights[i])):
                hidden_error[i][j] = out_error_derivative[j] * self.output_weights[i][j]

        #sum hidden errors to combine for layer 1
        for i in range(self.num_hidden):
            for j in range(len(hidden_error[i])):
                hidden_error_summed[j] += hidden_error[i][j]

        #calculate hidden error * sigmoid derivative
        for i in range(self.num_hidden):
            input_error[i] = hidden_error_summed[i] * self.sigmoid_derivative(out_h[i])

        #calculate the input -> hidden error
        for i in range(self.num_input):
            for j in range(len(input_error)):
                input_hidden_grad[i][j] = input_error[j] * inputs[i]

        #calculate hidden layer bias gradient
        for i in range(len(self.hidden_bias)):
            hidden_bias_grad[i] = self.sigmoid_derivative(out_h[i]) * input_error[i]

        #calculate output layer bias gradient
        for i in range(len(self.output_bias)):
            output_bias_grad[i] = self.sigmoid_derivative(out_o[i]) * error[i]
        return hidden_output_grad, input_hidden_grad, hidden_bias_grad, output_bias_grad

args = sys.argv

inputs = args[1]
hiddens = args[2]
outputs = args[3]
acc = []


# train_pickle = open("trainDigitX.pickle", "rb")
# train_pickle_out = open("trainDigitY.pickle", "rb")
# test_pickle = open("testDigitX.pickle", "rb")
# test_pickle_out = open("testDigitY.pickle", "rb")
# test_pickle_2 = open("testDigitX2.pickle", "rb")

# training_data = pickle.load(train_pickle)
# out = pickle.load(train_pickle_out)
# test_data = pickle.load(test_pickle)
# test_data_out = pickle.load(test_pickle_out)
# next_test_data = pickle.load(test_pickle_2)

training_data = np.loadtxt(args[4], dtype=float, delimiter=',')
out = np.loadtxt(args[5], dtype=float)
test_data = np.loadtxt(args[6], dtype=float, delimiter=',')
test_data_out = np.loadtxt(args[7], dtype=float)
next_test_data = np.loadtxt(args[8], dtype=float, delimiter=',')

output_data = []
av_plot = []

for i in range(len(out)):
    newo = [0 for j in range(10)]
    newo[int(out[i])] = 1
    output_data.append(newo)

network = NeuralNetwork(inputs, hiddens, outputs)
num_batchs = len(training_data) // network.batch_size

for epoch in range(network.n_epochs):
    for batch in range(num_batchs):
        print("batch number: ", batch)

        current = training_data[(batch * network.batch_size): ((batch + 1) * network.batch_size)]
        current_target = output_data[(batch * network.batch_size): ((batch + 1) * network.batch_size)]

        current_size = len(current)

        #initialise gradients sum and average arrays
        hidden_weights_sum = [[0 for i in range(network.num_hidden)] for j in range(network.num_input)]
        output_weights_sum = [[0 for i in range(network.num_output)] for j in range(network.num_hidden)]
        hidden_bias_sum = [0 for i in range(network.num_hidden)]
        output_bias_sum = [0 for i in range(network.num_output)]

        for i in range(len(current)):
            out_h, out_o, error = network.forward_pass(current[i], current_target[i])
            out_g, hidden_g, hbias_g, obias_g = network.backward_pass(out_h, out_o, error, current[i], current_target[i])

            #add gradients from back pass to sum total
            for i in range(len(out_g)):
                for j in range(len(out_g[i])):
                    output_weights_sum[i][j] += out_g[i][j]

            for i in range(len(hidden_g)):
                for j in range(len(hidden_g[i])):
                    hidden_weights_sum[i][j] += hidden_g[i][j]

            for i in range(len(hbias_g)):
                hidden_bias_sum[i] += hbias_g[i]

            for i in range(len(obias_g)):
                output_bias_sum[i] += obias_g[i]

        #update weights at end of mini batch
        network.update_weights(output_weights_sum, hidden_weights_sum, hidden_bias_sum, output_bias_sum, current_size)
        np.savetxt('HiddenWeights.csv', network.hidden_weights)
        np.savetxt('OutputWeights.csv', network.output_weights)
        np.savetxt('HiddenBias.csv', network.hidden_bias)
        np.savetxt('OutputBias.csv', network.output_bias)

    errorSum = 0
    for i in range(len(error)):
        errorSum += error[i]
    av_error = network.cost_function(errorSum)
    av_plot.append(av_error)

    predictions = network.test_data(test_data, test_data_out)
    # right = 0
    # for i in range(len(predictions)):
    #     if predictions[i] == test_data_out[i]:
    #         right += 1
    # accuracy = (right / len(test_data_out) * 100)
    # print("accuracy = ", accuracy, "%")
    # acc.append(accuracy)

    # next_predictions = network.predict_data(next_test_data)
# np.savetxt('PredictDigitY2.csv.gz', next_predictions, fmt='%0f')
plt.plot(av_plot)
plt.ylabel('Cost function')
plt.xlabel('Epoch')
plt.show()


        

        


