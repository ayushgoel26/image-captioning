import numpy as np
from networks.CNN.Convolution import Convolution
from networks.CNN.Flatten import Flatten
from networks.CNN.Maxpool import Maxpool
from TrainData import TrainData
import torch.nn as nn
import torch.optim as optim
from conf import MAX_POOL_K_SIZE, MAX_POOL_PADDING, MAX_POOL_STRIDE, CONVOLUTION_K_SIZE, CONVOLUTION_PADDING, \
    CONVOLUTION_STRIDE, OUT_CHANNEL, IN_CHANNEL, LEARNING_RATE, INPUT_DIMENSION_RNN, HIDDEN_DIMENSION_RNN

class CNN:
    def __init__(self):
        input_shapes = [[IN_CHANNEL[0], 112, 112], [IN_CHANNEL[1], 56, 56], [IN_CHANNEL[2], 28, 28],
                        [IN_CHANNEL[3], 14, 14], [IN_CHANNEL[4], 7, 7]]

        self.convolution_layer_1 = Convolution(in_channels=IN_CHANNEL[0], out_channels=OUT_CHANNEL[0],
                                               kernel_size=CONVOLUTION_K_SIZE, padding=CONVOLUTION_PADDING,
                                               stride=CONVOLUTION_STRIDE, input_shape=input_shapes[0])
        self.max_pool_layer_1 = Maxpool(MAX_POOL_K_SIZE, stride=MAX_POOL_STRIDE, padding=MAX_POOL_PADDING)

        self.convolution_layer_2 = Convolution(in_channels=IN_CHANNEL[1], out_channels=OUT_CHANNEL[1],
                                               kernel_size=CONVOLUTION_K_SIZE, padding=CONVOLUTION_PADDING,
                                               stride=CONVOLUTION_STRIDE, input_shape=input_shapes[1])
        self.max_pool_layer_2 = Maxpool(MAX_POOL_K_SIZE, stride=MAX_POOL_STRIDE, padding=MAX_POOL_PADDING)

        self.convolution_layer_3 = Convolution(in_channels=IN_CHANNEL[2], out_channels=OUT_CHANNEL[2],
                                               kernel_size=CONVOLUTION_K_SIZE, padding=CONVOLUTION_PADDING,
                                               stride=CONVOLUTION_STRIDE, input_shape=input_shapes[2])
        self.max_pool_layer_3 = Maxpool(MAX_POOL_K_SIZE, stride=MAX_POOL_STRIDE, padding=MAX_POOL_PADDING)

        self.convolution_layer_4 = Convolution(in_channels=IN_CHANNEL[3], out_channels=OUT_CHANNEL[3],
                                               kernel_size=CONVOLUTION_K_SIZE, padding=CONVOLUTION_PADDING,
                                               stride=CONVOLUTION_STRIDE, input_shape=input_shapes[3])
        self.max_pool_layer_4 = Maxpool(MAX_POOL_K_SIZE, stride=MAX_POOL_STRIDE, padding=MAX_POOL_PADDING)

        self.convolution_layer_5 = Convolution(in_channels=IN_CHANNEL[4], out_channels=OUT_CHANNEL[4],
                                               kernel_size=CONVOLUTION_K_SIZE, padding=CONVOLUTION_PADDING,
                                               stride=CONVOLUTION_STRIDE, input_shape=input_shapes[4])
        self.rnn = nn.LSTM(INPUT_DIMENSION_RNN, HIDDEN_DIMENSION_RNN, batch_first=True)
        self.flatten = Flatten()
        self.learning_rate = LEARNING_RATE
        self.training_data = TrainData()

    def forward(self, image_vector):
        conv_out = self.convolution_layer_1.forward(image_vector)
        conv_out = np.resize(conv_out, (conv_out.shape[1], conv_out.shape[2], conv_out.shape[0]))
        max_pool_out = self.max_pool_layer_1.forward(conv_out)
        max_pool_out = np.reshape(max_pool_out, (max_pool_out.shape[2], max_pool_out.shape[0], max_pool_out.shape[1]))

        conv_out = self.convolution_layer_2.forward(max_pool_out)
        conv_out = np.resize(conv_out, (conv_out.shape[1], conv_out.shape[2], conv_out.shape[0]))
        max_pool_out = self.max_pool_layer_2.forward(conv_out)
        max_pool_out = np.reshape(max_pool_out, (max_pool_out.shape[2], max_pool_out.shape[0], max_pool_out.shape[1]))

        conv_out = self.convolution_layer_3.forward(max_pool_out)
        conv_out = np.resize(conv_out, (conv_out.shape[1], conv_out.shape[2], conv_out.shape[0]))
        max_pool_out = self.max_pool_layer_3.forward(conv_out)
        max_pool_out = np.reshape(max_pool_out, (max_pool_out.shape[2], max_pool_out.shape[0], max_pool_out.shape[1]))

        conv_out = self.convolution_layer_4.forward(max_pool_out)
        conv_out = np.resize(conv_out, (conv_out.shape[1], conv_out.shape[2], conv_out.shape[0]))
        max_pool_out = self.max_pool_layer_4.forward(conv_out)
        max_pool_out = np.reshape(max_pool_out, (max_pool_out.shape[2], max_pool_out.shape[0], max_pool_out.shape[1]))
        cnn_out = self.flatten.forward(self.convolution_layer_5.forward(max_pool_out))

        lstm_out, hidden = self.rnn(input=[], hidden=cnn_out)
        return lstm_out, hidden

    def backward(self, compressed_image_vector):
        derivative = self.convolution_layer_5.backward(self.flatten.backward(compressed_image_vector),
                                                       self.learning_rate)
        derivative = np.reshape(derivative, (derivative.shape[1], derivative.shape[2], derivative.shape[0]))
        derivative = self.max_pool_layer_4.backward(derivative)
        derivative = np.reshape(derivative, (derivative.shape[2], derivative.shape[0], derivative.shape[1]))
        derivative = self.convolution_layer_4.backward(derivative,self.learning_rate)
        derivative = np.reshape(derivative, (derivative.shape[1], derivative.shape[2], derivative.shape[0]))
        derivative = self.max_pool_layer_3.backward(derivative)
        derivative = np.reshape(derivative, (derivative.shape[2], derivative.shape[0], derivative.shape[1]))
        derivative = self.convolution_layer_3.backward(derivative,self.learning_rate)
        derivative = np.reshape(derivative, (derivative.shape[1], derivative.shape[2], derivative.shape[0]))
        derivative = self.max_pool_layer_2.backward(derivative)
        derivative = np.reshape(derivative, (derivative.shape[2], derivative.shape[0], derivative.shape[1]))
        derivative = self.convolution_layer_2.backward(derivative,self.learning_rate)
        derivative = np.reshape(derivative, (derivative.shape[1], derivative.shape[2], derivative.shape[0]))
        derivative = self.max_pool_layer_1.backward(derivative)
        derivative = np.reshape(derivative, (derivative.shape[2], derivative.shape[0], derivative.shape[1]))
        self.convolution_layer_1.backward(derivative, self.learning_rate)

    def train(self, data, iterations):
        error = 0
        for iteration in range(iterations):
            for key in data:
                predicted_output = self.forward(data[key]['image'])
                error += self.training_data.binary_cross_entropy(y, predicted_output)
                self.backward(self.training_data.binary_cross_entropy_prime(y, predicted_output))

    def test(self):
        pass
