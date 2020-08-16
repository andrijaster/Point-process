import torch
from torch import nn


class LSTMPointProcess(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=2,
                 n_layers=2, dropout=0, num_layers=2, step=2):
        super(LSTMPointProcess, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers,
                           dropout = dropout, batch_first=True)
        lst = nn.ModuleList()
        # Fully connected layer
        out_size = hidden_dim
        for i in range(num_layers):
            inp_size = out_size
            out_size = int(inp_size//step)
            if i == num_layers-1:
                block = nn.Linear(inp_size, 1)
            else:
                block = nn.Sequential(nn.Linear(inp_size, out_size),
                                      nn.ReLU())
            lst.append(block)

        self.fc = nn.Sequential(*lst)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

    def forward(self, x, t, hidden = None):
        t = t.to(torch.float32)
        x = torch.cat((x,t.reshape(1,-1)),dim = 1)
        x = x.reshape(1,1,-1)

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        if out.shape[0] == 1:
            out = out.contiguous().view(-1, self.hidden_dim)
            out = torch.exp(self.fc(out))
        else: out = torch.transpose(torch.exp(self.fc(out)).squeeze(),0,1)

        return out, hidden
