import torch as T
import torch.nn as nn

class SimpleRNN(nn.Module):
    """ A very simple RNN cell with a single hidden vector """
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(SimpleRNN, self).__init__()

        self.n_hidden = n_hidden

        n_comb = n_inputs + n_hidden
        self.in_to_hid  = nn.Linear( n_comb, n_hidden )
        self.in_to_out  = nn.Linear( n_comb, n_outputs )

    def forward(self, input, hidden):
        combined = T.cat( ( input,hidden), 1 )
        hidden = self.in_to_hid( combined )
        output = self.in_to_out( combined )
        return output, hidden

    def reset_hidden(self):
        return T.zeros(1, self.n_hidden)
