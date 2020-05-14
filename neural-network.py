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
        self.hiddenWeights = [[np.random.rand()/2 for i in range(num_hidden)] for j in range(num_input)]
        self.outputWeights = [[np.random.rand()/2 for i in range(num_output)] for j in range(num_hidden)]
        self.hidden_bias = [np.random.rand()/2 for i in range(num_hidden)]
        self.output_bias = [np.random.rand()/2 for i in range(num_output)]
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
        return (1/(2*self.num_input)) * ((x - y)**2)

    def update_weights(self, hidden_gradients, output_gradients, hidden_bias_grad, output_bias_grad):
        ind = 0
        for i in range(self.num_hidden):
            for j in range(self.num_input):
                self.hiddenWeights[j][i] -= (self.learning_rate * hidden_gradients[ind])
                ind += 1   
        ind = 0
        for i in range(self.num_output):
            for j in range(self.num_hidden):
                self.outputWeights[j][i] -= (self.learning_rate * output_gradients[ind])
                ind += 1
 
        for i in range(len(self.hidden_bias)):
            self.hidden_bias[i] -= (self.learning_rate * hidden_bias_grad[i])

        for i in range(len(network.output_bias)):
            self.output_bias[i] -= (self.learning_rate * output_bias_grad[i])

    def predict(self, sample):
        outputL = [0 for i in range(self.num_output)]
        hiddenL = [0 for i in range(self.num_hidden)]

        for i in range(self.num_input):
            for j in range(self.num_hidden):
                hiddenL[j] = hiddenL[j] + (sample[i] * self.hiddenWeights[i][j])
        for i in range(self.num_hidden):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])

        for i in range(self.num_hidden):
            for j in range(self.num_output):
                outputL[j] = outputL[j] + (hiddenL[i] * self.outputWeights[i][j])
        for i in range(self.num_output):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        return outputL


    def forward_pass(self, sample, target):
        outputL = [0 for i in range(self.num_output)]
        hiddenL = [0 for i in range(self.num_hidden)]
        error = [0 for i in range(self.num_output)]

        for i in range(self.num_input):
            for j in range(self.num_hidden):
                hiddenL[j] = hiddenL[j] + (sample[i] * self.hiddenWeights[i][j])
        for i in range(self.num_hidden):
            hiddenL[i] = self.sigmoid(hiddenL[i] + self.hidden_bias[i])

        for i in range(self.num_hidden):
            for j in range(self.num_output):
                outputL[j] = outputL[j] + (hiddenL[i] * self.outputWeights[i][j])
        for i in range(self.num_output):
            outputL[i] = self.sigmoid(outputL[i] + self.output_bias[i])
        # print("out_o: ", outputL)
        # print("target: ", target)

        for i in range(len(error)):
            error[i] = self.cost_function(target[i], outputL[i])
        return hiddenL, outputL, error


    def backward_pass(self, out_h, out_o, inputs, target):
        weight_derivs = []
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
                deriv = error_outh[j] * self.sigmoid_derivative(out_h[j]) * inputs[i]
                weight_derivs.append(deriv)
        
        for i in range(self.num_hidden):
            for j in range(len(self.outputWeights[i])):
                deriv = -(target[j] - out_o[j]) * self.sigmoid_derivative(out_o[j]) * (out_h[i])
                weight_derivs.append(deriv)


        for i in range(len(self.hidden_bias)):
                deriv = self.sigmoid_derivative(out_h[i]) * error_outh[i]
                weight_derivs.append(deriv)

        for i in range(len(self.output_bias)):
                deriv = self.sigmoid_derivative(out_o[i]) * error_outh[i]
                weight_derivs.append(deriv)
        return weight_derivs


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
        # print("batch number: ", batch)

        current = training_data[(batch * network.batch_size): ((batch + 1) * network.batch_size)]
        current_target = output_data[(batch * network.batch_size): ((batch + 1) * network.batch_size)]

        current_size = len(current)

        #initialise gradients sum and average arrays
        gradients_sum = [0 for i in range((inputs * hiddens) + (hiddens * outputs) + hiddens + outputs)]
        average_gradients = [0 for i in range((inputs * hiddens) + (hiddens * outputs) + hiddens + outputs)]

        for i in range(len(current)):
            out_h, out_o, error = network.forward_pass(current[i], current_target[i])
            gradients = network.backward_pass(out_h, out_o, current[i], current_target[i])

            #add gradients from back pass to sum total
            for i in range(len(gradients)):
                gradients_sum[i] += gradients[i]

        #get the average gradients from the batch
        for i in range(len(average_gradients)):
            average_gradients[i] = gradients_sum[i] / current_size
        
        hidden_gradients = average_gradients[: (network.num_input * network.num_hidden)]
        output_gradients = average_gradients[(network.num_input * network.num_hidden) : (network.num_input * network.num_hidden) + (network.num_output * network.num_hidden)]
        hidden_bias_grad = average_gradients[(network.num_input * network.num_hidden) + (network.num_output * network.num_hidden): (network.num_input * network.num_hidden) + (network.num_output * network.num_hidden) + network.num_hidden]
        output_bias_grad = average_gradients[(network.num_input * network.num_hidden) + (network.num_output * network.num_hidden) + network.num_hidden : (network.num_input * network.num_hidden) + (network.num_output * network.num_hidden) + network.num_hidden + network.num_output]

        network.update_weights(hidden_gradients, output_gradients, hidden_bias_grad, output_bias_grad)
        
        

        errorSum = 0
        for i in range(len(error)):
            errorSum += error[i]
        av_error = errorSum / len(error)
        av_plot.append(av_error)
        
        plt.plot(av_plot)
        plt.ylabel('Error')
    plt.show()

        # print("hidden weights: ", network.hiddenWeights)
        # print("output weights: ", network.outputWeights)
        # print("hidden bias: ", network.hidden_bias)
        # print("output bias: ", network.output_bias)


# read in test data
# forward feed sample through network
# get the highest value of the output matrix, return the index as the prediction

for i in range(len(test_data)):
    predict = network.forward_pass(test_data[i])
    max = 0
    for j in range(len(predict)):
        if predict[j] > predict[max]:
            max = j

    print("prediction for sample %: ", i, max, "    actual value is: %", test_data_out[i])
