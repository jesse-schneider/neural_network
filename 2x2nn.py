import math
import numpy as np
import pickle

class NeuralNetwork():
    def __init__(self, num_input, num_hidden, num_output):
        #initalise network
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.hidden_weights = [[0.1, 0.2],[0.1, 0.1]]
        self.output_weights = [[0.1, 0.1], [0.1, 0.2]]
        self.hidden_bias = [0.1, 0.1]
        self.output_bias = [0.1, 0.1]
        self.learning_rate = 3
        self.n_epochs = 3
        self.batch_size = 2
        

    #sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    #sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def cost_function(self, x, y):
       return (1/2*self.num_input) * ((x - y)**2)

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


    def update_weights(self, output_gradients, hidden_gradients, hidden_bias_grad, output_bias_grad):
        
        #update hidden -> output weights
        for i in range(self.num_hidden):
            for j in range(self.num_output):
                self.output_weights[i][j] -= (self.learning_rate * (output_gradients[i][j] / 2))

        #update input -> hidden weights
        for i in range(self.num_input):
            for j in range(self.num_hidden):
                self.hidden_weights[i][j] -= (self.learning_rate * (hidden_gradients[i][j] / 2))
 
        #update hidden bias
        for i in range(len(self.hidden_bias)):
            self.hidden_bias[i] -= (self.learning_rate * (hidden_bias_grad[i] / 2))

        #update output bias
        for i in range(len(network.output_bias)):
            self.output_bias[i] -= (self.learning_rate * (output_bias_grad[i] / 2))

    def predict(self, sample):
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
        outputL = np.zeros(self.num_output)
        hiddenL = np.zeros(self.num_hidden)
        error = np.zeros(self.num_output)

        #compute all inputs x weights into each hidden node
        for i in range(self.num_input):
            for j in range(self.num_hidden):
                hiddenL[j] = hiddenL[j] + (sample[i] * self.hidden_weights[i][j])
        for i in range(self.num_hidden):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])

        #compute all hidden x weights into each output node
        for i in range(self.num_hidden):
            for j in range(self.num_output):
                outputL[j] = outputL[j] + (hiddenL[i] * self.output_weights[i][j])
        for i in range(self.num_output):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        # print(outputL)

        #calculate each output node error
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

training_data = [[0.1, 0.1], [0.1, 0.2]]
output_data = [[1, 0], [0, 1]]

inputs = 2
hiddens = 2
outputs = 2

network = NeuralNetwork(inputs, hiddens, outputs)
num_batchs = len(training_data) // network.batch_size

for epoch in range(network.n_epochs):

    #initialise gradients sum and average arrays
    hidden_weights_sum = [[0 for i in range(network.num_hidden)] for j in range(network.num_input)]
    output_weights_sum = [[0 for i in range(network.num_output)] for j in range(network.num_hidden)]
    hidden_bias_sum = [0 for i in range(network.num_hidden)]
    output_bias_sum = [0 for i in range(network.num_output)]

    gradients_sum = [0 for i in range((inputs * hiddens) + (hiddens * outputs) + hiddens + outputs)]
    average_gradients = [0 for i in range((inputs * hiddens) + (hiddens * outputs) + hiddens + outputs)]

    for i in range(len(training_data)):
        out_h, out_o, error = network.forward_pass(training_data[i], output_data[i])
        out_g, hidden_g, hbias_g, obias_g = network.backward_pass(out_h, out_o, error, training_data[i], output_data[i])

        print("out_gradient: ", out_g)
        print("hidden_gradient: ", hidden_g)
        print("hidden_bias_gradient: ", hbias_g)
        print("out_bias_gradient: ", obias_g)

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
    network.update_weights(output_weights_sum, hidden_weights_sum, hidden_bias_sum, output_bias_sum)

    # print("hidden weights: ", network.hidden_weights)
    # print("output weights: ", network.output_weights)
    # print("hidden bias: ", network.hidden_bias)
    # print("output bias: ", network.output_bias)
