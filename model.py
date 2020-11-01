import torch


class NameGenerationRNN(torch.nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(NameGenerationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(n_categories + input_size, hidden_size)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size)
        self.h2l = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, category, input, hidden):
        '''
        function implementing steps of forward pass through NN
        :param category: one-hot tensor for category of surname
        :param input: one-hot tensor of <SOS> token or previous output based on which next letter is predicted
        :param hidden: previous state tensor, zero tensor if input is <SOS>
        :return: output tensor of log probabilities over all letters and hidden state computed at current step
        '''
        input_combined = torch.cat((category, input), dim=1)
        h1 = self.i2h(input_combined)
        h2 = self.h2h(hidden)
        hidden = h1 + h2
        h3 = self.h2l(hidden)
        output = self.dropout(h3)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


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
