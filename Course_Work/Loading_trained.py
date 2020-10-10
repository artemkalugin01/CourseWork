import numpy as np
import random
import pandas as pd
import re
import json
import math

random.seed(0)
np.random.seed(0)


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, inputs, neurons, weight_regularizer_l1=0,
                 weight_regularizer_l2=0., bias_regularizer_l1=0, bias_regularizer_l2=0.):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        hgfjv = 0

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = self.weights.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = self.biases.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


# Dropout
class Layer_Dropout:

    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, values):
        # Save input values
        self.input = values
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=values.shape) / self.rate
        # Apply mask to output values
        self.output = values * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dvalues = dvalues * self.binary_mask


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        dvalues = dvalues.copy()  # Since we need to modify original variable, let;s make a copy of values first
        dvalues[self.inputs <= 0] = 0  # Zero gradient where input values were negative
        self.dvalues = dvalues


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.dvalues = dvalues


# Common loss class
class Loss:

    # Regularization loss calculation
    def regularization_loss(self, layer):

        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0:  # only calculate when factor greaten than 0
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:  # only calculate when factor greater than 0
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.weights * layer.weights)

        return regularization_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(samples), y_true]

        # Losses
        ############################
        # вручную сумму считать
        summ = 0.
        for value in y_pred:
            summ += value

        ############################
        negative_log_likelihoods = -np.log10(summ)

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true

        # Overall loss

        data_loss = negative_log_likelihoods / samples
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        dvalues = dvalues.copy()  # We need to modify variable directly, make a copy first then
        dvalues[range(samples), y_true] -= 1
        dvalues = dvalues / samples

        self.dvalues = dvalues


# SGD Optimizer
class Optimizer_SGD:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., momentum=0., nesterov=False):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.nesterov = nesterov

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain momentum arrays, create ones filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # If we use momentum
        if self.momentum:

            # Build weight updates with momentum - take previous updates multiplied by retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

            # Apply Nesterov as well?
            if self.nesterov:
                weight_updates = self.momentum * weight_updates - self.current_learning_rate * layer.dweights
                bias_updates = self.momentum * bias_updates - self.current_learning_rate * layer.dbiases

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights with updates which are either vanilla, momentum or momentum+nesterov updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Adagrad Optimizer
class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# RMSprop Optimizer
class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Adam Optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-8, beta_1=0.999, beta_2=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (
                self.iterations + 1))  # self.iteration is 0 at first pass ans we need to start with 1 here
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        # Get corrected bias
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


def time_to_minutes(value_list):
    newarr = []
    for val in value_list:
        nums = re.split(' |;', val)
        time = nums[1].split(':')
        minutes = float(int(time[0]) * 60 + int(time[1]))
        newarr.append(minutes)
    return newarr


def read_float_with_comma(num): return float(str(num).replace(",", "."))


def td_to_float(value_list):
    new_arr = []
    for val in value_list:
        new_arr.append(read_float_with_comma(val) + 0.1)
    return new_arr


def standartization(value_list):
    newarr = []
    maximum = 10 ** -5
    for val in value_list:
        if val > maximum: maximum = val
    for val in value_list:
        newarr.append(float(val) / maximum)
    return newarr


# метод для преобразования IP адресов
def coher_of_values(column):
    newarr = []
    values = column.values
    unique_list = column.unique()
    unique_list = unique_list.tolist()
    for val in values:
        newarr.append(unique_list.index(val))
    return newarr


# Обработка Данных

X = pd.read_csv('/Users/artemkalugin/Desktop/Fixed_Partial_Dataset_1.csv', sep=',')
X.to_csv('test.csv')
aa = pd.read_csv('test.csv',sep=',')
print(12345678765432)
print(aa.head())
print(12345676543)
y = X['res']
X = X.drop(['res'], axis=1)
X = X.drop(['sa','da','Unnamed: 0'],axis=1)

filename = 'dences.txt'
myfile = open(filename,mode='r+')
dences =json.load(myfile)
from_json1 = dences[0]
from_json2 = dences[1]


dense1 = Layer_Dense(10, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
dense1.weights = from_json1['weights']
dense1.biases = from_json1['biases']

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create dropout layer
dropout1 = Layer_Dropout(0)

# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 2)  # second dense layer, 3 inputs, 3 outputs
dense2.weights = from_json2['weights']
dense2.biases = from_json2['biases']

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Create optimizer
# optimizer = Optimizer_SGD(decay=1e-8, momentum=0.9)
#optimizer = Optimizer_Adagrad(decay=1e-8)
optimizer = Optimizer_RMSprop(learning_rate=0.05, decay=4e-8, rho=0.999)
#optimizer = Optimizer_Adam(learning_rate=0.05, decay=1e-8)

# Make a forward pass of our training data thru this layer
dense1.forward(X)

# Make a forward pass thru activation function - we take output of previous layer here
activation1.forward(dense1.output)

# Make a formward pass thru second Dense layer - it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass thru activation function - we take output of previous layer here
activation2.forward(dense2.output)

# Calculate loss from output of activation2 so softmax activation
loss = loss_function.forward(activation2.output, y)

# Calculate accuracy from output of activation2 and targets
predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
accuracy = np.mean(predictions == y)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')





