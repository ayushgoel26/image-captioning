import torch


class TrainData:
    """
    Training CNN data MNIST
    """
    def __init__(self):
        pass

    @staticmethod
    def binary_cross_entropy_prime(y_true, y_pred):
        # y_pred is a vector of size 25088
        # y_true is just a 0 or 1
        y_pred = torch.Tensor(y_pred)

        # if y_true is 1, making it an array of ones else zeros
        if y_true:
            y_true = torch.ones_like(y_pred)
        else:
            y_true = torch.zeros_like(y_pred)  

        # big formula
        a = torch.empty
        b = torch.empty
        a = torch.div(torch.sub(y_true, torch.Tensor(1)),torch.sub(y_pred, torch.Tensor(1)))
        b = torch.div(y_true,y_pred)

        return torch.div(torch.sub(a, b), torch.Tensor(y_true.size())).numpy()

    @staticmethod
    def get_train_data():
        # read tensor data for x_train and y_train from .csv in
        x_train = torch.load('src/data/MNIST_training/mnist_x_train.csv')
        y_train = torch.load('src/data/MNIST_training/mnist_y_train.csv')

        # convert x_train and y_train to tensors
        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)

        # 60000 MNIST images
        # shape of x_train is 60000 * 28 * 28
        # each of 60k images have 28 * 28 dimensions

        # Initilizing output tensor
        x_train_dims_increased = torch.empty((1000,3,112,112))  # picking 1000 images of 3 * 112 * 112

        # we need only 1000 images as 60000 will take time
        # convert x_train and y_train to tensors
        # x_train = torch.tensor(x_train)
        # y_train = torch.tensor(y_train)
        y_train = y_train.resize_(1000)

        # x_train shape is 60000 * (28 * 28)
        # y_train shape is 60000
        # increasing the shape to 60000 * (3 * 112 * 112)

        # Iteraring through 1000 images
        for index in range(0, 1000):
            # index refers to each image
            temp = x_train[index].flatten()     # after flatten the dimensions are 28 * 28

            # for increasing dimensions, we are appending to dummy zero array
            # want a dummy 0 array of size (3 * 112 * 112) - (28 * 28) as we want to append 28 * 28 flattened array
            dummy_zero = torch.zeros(3*112*112 - (28*28))
            expanded_array = torch.cat((dummy_zero, temp))
            expanded_array = torch.reshape(expanded_array, (1, 3, 112, 112))
            x_train_dims_increased[index] = expanded_array

        return x_train_dims_increased.numpy(), y_train.numpy()

