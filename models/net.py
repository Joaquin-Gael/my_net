import numpy as np
#from fontTools.varLib.instancer import setRibbiBits
import matplotlib.pyplot as plt
import models.activation as ac
import models.loss_function as lf
import models.optimizer as op

class DataHandler:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def batch(self, batch_size):
        indices = np.random.permutation(len(self.x_data))
        for i in range(0, len(self.x_data), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield self.x_data[batch_indices], self.y_data[batch_indices]


class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.weights_matriz = np.random.randn(input_size, output_size) * np.sqrt(2 / output_size)
        #print('pesos shape: ',self.weights_matriz.shape)
        self.bias_vector = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        match self.activation:
            case 'relu':
                self.activation = ac.ReLu()

            case 'sig':
                self.activation = ac.Sigmoid()

        self.z = np.dot(self.inputs, self.weights_matriz) + self.bias_vector
        self.a = self.activation.activate(self.z)

        return self.a

    def backward(self, dA, learning_rate: float = 0.1):
        dA_resized = np.resize(dA, self.activation.derivative(self.z).shape)
        dZ = self.activation.derivative(self.z) * dA_resized
        dW = np.dot(self.inputs.T, dZ) / self.inputs.shape[0]
        dB = np.sum(dZ, axis=0, keepdims=True) / self.inputs.shape[0]
        if self.weights_matriz.shape == dW.shape:
            self.weights_matriz -= learning_rate * dW
        else:
            print("Shapes do not match. Adjusting dW...")
            dW = dW[:self.weights_matriz.shape[0], :self.weights_matriz.shape[1]]
            self.weights_matriz -= learning_rate * dW

        self.bias_vector -= dB * learning_rate

        return np.dot(dZ, self.weights_matriz.T)


class NeuronalNet:
    def __init__(self):
        self.layers_list: list[Layer] = []
        self.loss_function = lf.LossFunction()
        self.optimizer = op.SGD()

    def add_layer(self, input_size: int, output_size: int, activation: str = 'relu'):
        #if not self.layers_list:
        self.layers_list.append(Layer(input_size, output_size, activation))
        #else:
        #    pass

    def compile(self, loss_f, optimizer_f):
        self.loss_function = loss_f
        self.optimizer = optimizer_f

    def set_lr(self, lr: float = 0.1):
        self.optimizer.learning_rate = lr

    def forward(self, inputs):
        for layer in self.layers_list:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, X, Y, learning_rate):
        y_pred = self.forward(X)
        loss = self.loss_function.mse(Y, y_pred)
        dA = self.loss_function.mse_derivative(Y, y_pred)
        for layer in self.layers_list:
            dA = layer.backward(dA, learning_rate)
        return loss

    def fit(self, X, Y, batch_size, epochs: int = 1000):
        data_handler = DataHandler(X, Y)
        loss_list = []
        for epoch in range(epochs):
            for x_batch, y_batch in data_handler.batch(batch_size):
                loss = self.backward(x_batch, y_batch, self.optimizer.learning_rate)
            loss_list.append(loss)
            if epoch & 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

        self.plot_loss(loss_list)

    def predict(self, inputs):
        output = self.forward(inputs)
        return output

    def plot_loss(self, loss_history):
        plt.plot(range(len(loss_history)), loss_history, label='Pérdida')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Gráfico de la Pérdida durante el Entrenamiento')
        plt.legend()
        plt.show()

if __name__ == '__main__':

    matriz_input = np.random.randn(4,2)
    matriz_label = np.random.randn(4,1)

    print(matriz_label)
    print(matriz_input)

    nn = NeuronalNet()

    nn.add_layer(2, 4, activation='sig')
    nn.add_layer(4, 4, activation='sig')
    nn.add_layer(4, 2)

    nn.fit(matriz_input, matriz_label, batch_size=1)