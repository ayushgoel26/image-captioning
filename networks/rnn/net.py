import torch
import torch.nn as nn
import torch.optim as optim


class RNN(object):
    def __init__(self, input_dim, hidden_rnn_dim):
        super(RNN, self).__init__()
        self.batch_size = None
        self.rnn = nn.LSTM(input_dim, hidden_rnn_dim, batch_first=True)

    def forward(self, input_rnn):
        lstm_out, hidden = self.rnn(input_rnn)
        return lstm_out, hidden

    def model_train(self, epochs, learning_rate, data):
        print('Training')
        loss_fn = torch.nn.CrossEntropyLoss()
        optimiser = optim.Adam(self.rnn.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            total_loss = 0.0
            for sent_len in data:
                optimiser.zero_grad()
                # y_pred, hidden_out = self.forward(data[sent_len]['captions'][0].float())
                y_pred, hidden_out = self.forward(torch.rand(25088))
                loss = loss_fn(torch.squeeze(y_pred, 1),
                               torch.squeeze(torch.squeeze(data[sent_len]['captions'][0], 1), 1))
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            if epoch % 5 == 0:
                print('epoch: %d, loss: %.3f' % (epoch, total_loss))
        print('Finished Training')


rnn = RNN(input_dim=25088, hidden_rnn_dim=30000)
from data import Processor
processor = Processor()
print("Processing Captions")
processor.caption_reader()
print("Captions processed")
rnn.model_train(1, 0.01, processor.data)
