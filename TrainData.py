from data.processing import Processor
import numpy as np
from torchvision import datasets
import torch


class TrainData:
    def __init__(self):
        pass

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_prime(y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    @staticmethod
    def getTrainData():
        training_data_size = 100
        input_dimensions = (3, 112, 112)
        output_training_dimensions = 512 * 7 * 7

        tmp = 2400
        data = datasets.MNIST('../data', train=True, download=True)
        train_loader = torch.utils.data.DataLoader(data)

        data_processing = Processor()
        x_train, y_train = data_processing.preprocessing_data_mnist(train_loader, tmp)
        x_train = np.reshape(x_train, np.append(training_data_size, input_dimensions))
        y_train = np.reshape(y_train, (1, y_train.shape[0] * y_train.shape[1]))
        y_train = np.resize(y_train, (1, output_training_dimensions))

        return x_train, y_train
