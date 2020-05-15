import math
import numpy as np
import pickle
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, num_input, num_hidden, num_output):
        #initalise network
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.hidden_weights = [[np.random.randn() for i in range(num_hidden)] for j in range(num_input)]
        self.output_weights = [[np.random.randn() for i in range(num_output)] for j in range(num_hidden)]
        self.hidden_bias = [np.random.randn() for i in range(num_hidden)]
        self.output_bias = [np.random.randn() for i in range(num_output)]
        self.learning_rate = 3
        self.n_epochs = 1
        self.batch_size = 20
        

    #sigmoid activation function
    def sigmoid(self, x):
        return (1.0 / (1.0 + math.exp(-x)))

    #sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def cost_function(self, x, y):
        return (1/2*self.num_input) * ((x - y)**2)

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
        outputL = [0 for i in range(self.num_output)]
        hiddenL = [0 for i in range(self.num_hidden)]
        error = [0 for i in range(self.num_output)]

        for i in range(self.num_input):
            for j in range(self.num_hidden):
                hiddenL[j] += sample[i] * self.hidden_weights[i][j]

        for i in range(self.num_hidden):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])

        for i in range(self.num_hidden):
            for j in range(self.num_output):
                outputL[j] = outputL[j] + (hiddenL[i] * self.output_weights[i][j])
        
        for i in range(self.num_output):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        print("out_o: ", outputL)
        # print("target: ", target)

        #calculate error from output node
        for i in range(len(error)):
            error[i] = (target[i] - outputL[i]) * self.sigmoid_derivative(outputL[i])
        return hiddenL, outputL, error


    def backward_pass(self, out_h, out_o, error, inputs, target):
        hidden_output_grad = [[0 for i in range(self.num_output)] for j in range(self.num_hidden)]
        input_hidden_grad = [[0 for i in range(self.num_hidden)] for j in range(self.num_input)]

        hidden_error = [[0 for i in range(self.num_output)] for j in range(self.num_hidden)]
        hidden_error_summed = [0 for i in range(self.num_hidden)]
        input_error = [0 for i in range(self.num_hidden)]
        hidden_bias_grad = [0 for i in range(self.num_hidden)]
        output_bias_grad = [0 for i in range(self.num_output)]


        #finding error in weights from hidden -> output
        for i in range(self.num_hidden):
            for j in range(len(error)):
                hidden_output_grad[i][j] = error[j] * out_h[i] * (-1)

        #find error caused by each weight in hidden layer
        for i in range(len(self.output_weights)):
            for j in range(len(self.output_weights[i])):
                hidden_error[i][j] = error[j] * self.output_weights[i][j]

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



inputs = 784
hiddens = 30
outputs = 10

train_pickle = open("trainDigitX.pickle", "rb")
train_pickle_out = open("trainDigitY.pickle", "rb")
test_pickle = open("testDigitX.pickle", "rb")
test_pickle_out = open("testDigitY.pickle", "rb")
training_data = pickle.load(train_pickle)
out = pickle.load(train_pickle_out)
test_data = pickle.load(test_pickle)
test_data_out = pickle.load(test_pickle_out)
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

        gradients_sum = [0 for i in range((inputs * hiddens) + (hiddens * outputs) + hiddens + outputs)]
        average_gradients = [0 for i in range((inputs * hiddens) + (hiddens * outputs) + hiddens + outputs)]

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
        
        

        errorSum = 0
        for i in range(len(error)):
            errorSum += error[i]
        av_error = errorSum / len(error)
        av_plot.append(av_error)
        
        plt.plot(av_plot)
        plt.ylabel('Error')
    plt.show()

        # print("hidden weights: ", network.hidden_weights)
        # print("output weights: ", network.output_weights)
        # print("hidden bias: ", network.hidden_bias)
        # print("output bias: ", network.output_bias)


# read in test data
# forward feed sample through network
# get the highest value of the output matrix, return the index as the prediction

for i in range(len(test_data)):
    predict = network.predict(test_data[i])
    max = 0
    for j in range(len(predict)):
        if predict[j] > predict[max]:
            max = j

    print("prediction for sample ", i, ":", max, "    actual value is:", test_data_out[i])
