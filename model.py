import torch


class NameGenerationRNN(torch.nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(NameGenerationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(n_categories + input_size, hidden_size)
        self.h1_to_h1 = torch.nn.Linear(hidden_size, hidden_size)
        self.h1_to_h2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.h2_to_h2 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.h2l = torch.nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dropout = torch.nn.Dropout(0.2)
        self.l2l = torch.nn.Linear(hidden_size // 4, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, category, input, hidden1, hidden2):
        '''
        function implementing steps of forward pass through NN
        :param category: one-hot tensor for category of surname
        :param input: one-hot tensor of <SOS> token or previous output based on which next letter is predicted
        :param hidden1: previous hidden-1 state tensor, zero tensor if input is <SOS>
        :param hidden2: previous hidden-2 state tensor, zero tensor if input is <SOS>
        :return: output tensor of log probabilities over all letters and hidden state computed at current step
        '''
        input_combined = torch.cat((category, input), dim=1)
        h1 = self.i2h(input_combined)
        h2 = self.h1_to_h1(hidden1)
        hidden1 = torch.relu(h1 + h2)
        h21 = self.h1_to_h2(hidden1)
        h22 = self.h2_to_h2(hidden2)
        hidden2 = torch.tanh(h21 + h22)
        h3 = self.h2l(hidden2)
        h3 = torch.relu(h3)
        h4 = self.dropout(h3)
        output = self.l2l(h4)
        output = self.softmax(output)
        return output, hidden1, hidden2

    def init_hidden1(self):
        return torch.zeros(1, self.hidden_size)

    def init_hidden2(self):
        return torch.zeros(1, self.hidden_size // 2)


class RNN(torch.nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = torch.nn.Linear(hidden_size + output_size, output_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
