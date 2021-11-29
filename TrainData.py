from data_processing import Processor

class TrainData:
      def __init__(self):
    pass

  def binary_cross_entropy(self, y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

  def binary_cross_entropy_prime(self, y_true, y_pred):
      return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

  def getTrainData(self):
      
      training_data_size = 100
      input_dimensions = (3,112,112)
      output_training_dimensions = 512*7*7

      tmp = 2400
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      data_processing = Processor()
      x_train, y_train = data_processing.preprocess_data_mnist(x_train, y_train, tmp)
      x_train = np.reshape(x_train, np.append(training_data_size, input_dimensions))
      y_train = np.reshape(y_train, (1,y_train.shape[0] * y_train.shape[1]))
      y_train = np.resize(y_train, (1,output_training_dimensions))

      return x_train, y_train