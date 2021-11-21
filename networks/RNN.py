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
        return lstm_out[:, -1, :]

    def model_train(self, epochs, learning_rate, data):
        print('Training')
        loss_fn = torch.nn.CrossEntropyLoss()
        optimiser = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            total_loss = 0.0
            for sent_len in data:
                optimiser.zero_grad()
                y_pred = self.forward(self.data[sent_len]['sentences'].float())
                loss = loss_fn(torch.squeeze(y_pred, 1),
                               torch.squeeze(torch.squeeze(self.data[sent_len]['labels'], 1), 1).long())
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            if epoch % 5 == 0:
                print('epoch: %d, loss: %.3f' % (epoch, total_loss))
        print('Finished Training')

    def test(self, sentence):
        y_pred = self.forward(sentence.float())
        print(self.out[torch.argmax(y_pred).item()])