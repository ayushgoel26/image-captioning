import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import torch

from TrainData import TrainData
from data.processing import Processor
from networks.cnn.convolution import Convolution
from networks.cnn.flatten import Flatten
from networks.cnn.maxpool import Maxpool
from conf import MAX_POOL_K_SIZE, MAX_POOL_PADDING, MAX_POOL_STRIDE, CONVOLUTION_K_SIZE, CONVOLUTION_PADDING, \
    CONVOLUTION_STRIDE, OUT_CHANNEL, IN_CHANNEL, LEARNING_RATE, INPUT_DIMENSION_RNN, HIDDEN_DIMENSION_RNN, CNN_PARAMETERS_FILE


class CaptionGenerator:
    def __init__(self, processor):
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
        self.linear = nn.Linear(HIDDEN_DIMENSION_RNN, INPUT_DIMENSION_RNN)
        self.flatten = Flatten()
        self.learning_rate = LEARNING_RATE
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.processor = processor

    def forward(self, image_vector):
        conv_out = self.convolution_layer_1.forward(image_vector)
        conv_out = np.reshape(conv_out, (conv_out.shape[1], conv_out.shape[2], conv_out.shape[0]))
        max_pool_out = self.max_pool_layer_1.forward(conv_out)
        max_pool_out = np.reshape(max_pool_out, (max_pool_out.shape[2], max_pool_out.shape[0], max_pool_out.shape[1]))

        conv_out = self.convolution_layer_2.forward(max_pool_out)
        conv_out = np.reshape(conv_out, (conv_out.shape[1], conv_out.shape[2], conv_out.shape[0]))
        max_pool_out = self.max_pool_layer_2.forward(conv_out)
        max_pool_out = np.reshape(max_pool_out, (max_pool_out.shape[2], max_pool_out.shape[0], max_pool_out.shape[1]))

        conv_out = self.convolution_layer_3.forward(max_pool_out)
        conv_out = np.reshape(conv_out, (conv_out.shape[1], conv_out.shape[2], conv_out.shape[0]))
        max_pool_out = self.max_pool_layer_3.forward(conv_out)
        max_pool_out = np.reshape(max_pool_out, (max_pool_out.shape[2], max_pool_out.shape[0], max_pool_out.shape[1]))

        conv_out = self.convolution_layer_4.forward(max_pool_out)
        conv_out = np.reshape(conv_out, (conv_out.shape[1], conv_out.shape[2], conv_out.shape[0]))
        max_pool_out = self.max_pool_layer_4.forward(conv_out)
        max_pool_out = np.reshape(max_pool_out, (max_pool_out.shape[2], max_pool_out.shape[0], max_pool_out.shape[1]))
        cnn_out = self.flatten.forward(self.convolution_layer_5.forward(max_pool_out))
        return cnn_out

    def backward(self, compressed_image_vector):
        derivative = self.convolution_layer_5.backward(self.flatten.backward(compressed_image_vector),
                                                       self.learning_rate)
        derivative = np.reshape(derivative, (derivative.shape[1], derivative.shape[2], derivative.shape[0]))
        derivative = self.max_pool_layer_4.backward(derivative)
        derivative = np.reshape(derivative, (derivative.shape[2], derivative.shape[0], derivative.shape[1]))
        derivative = self.convolution_layer_4.backward(derivative, self.learning_rate)
        derivative = np.reshape(derivative, (derivative.shape[1], derivative.shape[2], derivative.shape[0]))
        derivative = self.max_pool_layer_3.backward(derivative)
        derivative = np.reshape(derivative, (derivative.shape[2], derivative.shape[0], derivative.shape[1]))
        derivative = self.convolution_layer_3.backward(derivative, self.learning_rate)
        derivative = np.reshape(derivative, (derivative.shape[1], derivative.shape[2], derivative.shape[0]))
        derivative = self.max_pool_layer_2.backward(derivative)
        derivative = np.reshape(derivative, (derivative.shape[2], derivative.shape[0], derivative.shape[1]))
        derivative = self.convolution_layer_2.backward(derivative, self.learning_rate)
        derivative = np.reshape(derivative, (derivative.shape[1], derivative.shape[2], derivative.shape[0]))
        derivative = self.max_pool_layer_1.backward(derivative)
        derivative = np.reshape(derivative, (derivative.shape[2], derivative.shape[0], derivative.shape[1]))
        self.convolution_layer_1.backward(derivative, self.learning_rate)

    def train_cnn(self, epochs):
        training_data = TrainData()
        x_train, y_train = training_data.getTrainData()
        count = 0
        print("Starting epoch 1")
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                predicted_output = self.forward(x)
                grad = training_data.binary_cross_entropy_prime(y, predicted_output)
                self.backward(grad)
                print(count)
                count += 1
            print("Epoch %d done" %epoch)

    def save_cnn_parameters(self):
        cnn_parameters = [self.convolution_layer_1, self.max_pool_layer_1, self.convolution_layer_2,
                          self.max_pool_layer_2, self.convolution_layer_3, self.max_pool_layer_3,
                          self.convolution_layer_4,self.max_pool_layer_4, self.convolution_layer_5]
        torch.save(cnn_parameters, CNN_PARAMETERS_FILE)

    def train_rnn(self, data, iterations):
        error = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        optimiser = optim.Adam(self.rnn.parameters(), lr=LEARNING_RATE)
        for iteration in range(iterations):
            optimiser.zero_grad()
            key = random.choice(list(data.keys()))
            print(key)
            caption = torch.stack([torch.Tensor(i) for i in random.choice(data[key]["captions"])]).unsqueeze(0)
            predicted_output, hidden_output = self.forward(data[key]['image'], caption)
            print(caption.shape)
            print(predicted_output.shape)
            loss = loss_fn(caption, predicted_output)
            print("loss calculated")
            loss.backward()
            print("RNN backward done")
            # self.backward(self.training_data.binary_cross_entropy_prime(caption.numpy(),
            #                                                             predicted_output.detach().numpy()))
            # print("CNN backward done")
            optimiser.step()
            print("Optimizing done")

    def generate_caption(self):
        image_features = self.forward(input)
        sentence = []
        pred_word_vec, hidden_out = self.rnn(Processor.get_word_embedding('startseq')
                                             (torch.from_numpy(image_features).unsqueeze(0).float(),
                                              torch.from_numpy(image_features).unsqueeze(0).float()))
        word = self.get_word(pred_word_vec)
        sentence.append(word)
        while word != 'endseq' or len(sentence) <= 25:
            pred_word_vec, hidden_out = self.rnn(pred_word_vec,
                                                 (torch.from_numpy(hidden_out).unsqueeze(0).float(),
                                                  torch.from_numpy(hidden_out).unsqueeze(0).float()))
            word = self.get_word(pred_word_vec)
            sentence.append(word)

    def get_word(self, word_vec):
        words = []
        word_vectors = []
        for k in self.processor.model.wv.vocab:
            words.append(k)
            word_vectors.append(torch.from_numpy(self.processor.model[k]))
        print(len(words))
        print(len(word_vectors))
        return []
