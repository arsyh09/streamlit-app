import numpy as np

class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inisialisasi bobot antara input layer dan hidden layer secara acak (Weight)
        #self.input_weights = np.random.normal(size=(self.input_size, self.hidden_size))
        # optional
        self.input_weights = np.random.uniform(-1, 1, size=(self.input_size, self.hidden_size))

        # Inisialisasi bias pada hidden layer secara acak
        #self.bias_hidden = np.random.normal(size=(1, self.hidden_size))
        # optional
        self.bias_hidden = np.random.normal(0, 0.01, size=(1, self.hidden_size))

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def linear_activation(self, x):
        return x

    def relu_activation(self, x):
        return np.maximum(0, x)

    def tanh_activation(self, x):
        return np.tanh(x)

    def train(self, X, y, activation):
        if activation == 'sigmoid':
            self.activation = self.sigmoid_activation
        elif activation == 'linear':
            self.activation = self.linear_activation
        elif activation == 'relu':
            self.activation = self.relu_activation
        elif activation == 'tanh':
            self.activation = self.tanh_activation
        else:
            raise ValueError("Invalid activation function.")

        # Hitung output pada hidden layer
        H_init = np.dot(X, self.input_weights) + self.bias_hidden
        hidden_layer = self.activation(H_init)

        # Hitung bobot pada output layer dengan moore penrose pseudoinverse
        self.output_weights = np.dot(np.linalg.pinv(hidden_layer), y)

        # hitung hasil prediksi dari proses training
        train_output = np.dot(hidden_layer, self.output_weights)

        return train_output

    def predict(self, X):
        # Hitung output pada hidden layer
        H_init = np.dot(X, self.input_weights) + self.bias_hidden
        hidden_layer = self.activation(H_init)

        # Hitung output pada output layer
        output_layer = np.dot(hidden_layer, self.output_weights)

        return output_layer