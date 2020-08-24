import os
import glob
import string
import unicodedata
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

class NameDataset:
    def __init__(self, filenames):
        self.filenames = filenames

        self.char_list = " " + string.ascii_letters + ".,;"
        self.n_letters = len(self.char_list)
        self.dataset_list = []
        self.language_list = []

        for filename in glob.glob(self.filenames):

            language = os.path.splitext(os.path.basename(filename))[0]
            self.language_list.append(language)

            names = self._read_lines( filename )

            for name in names:
                self.dataset_list.append( [language, name] )

        self.n_languages = len(self.language_list)

    def _read_lines(self, filename):
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        return [ self._unicode_to_ascii(line) for line in lines ]

    def _unicode_to_ascii(self, s):
        return "".join( c for c in unicodedata.normalize("NFD", s)
                        if unicodedata.category(c) != "Mn"
                        and c in self.char_list
                      )

    def _word_2_idx_tensor( self, word ):
        as_list = [ self.char_list.index(char) for char in word ]
        as_tensor = T.tensor( as_list, dtype=T.int64 )
        return as_tensor

    def __len__(self):
        return len(self.dataset_list)

    def _get_rnd_batch(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)

        targets = []
        inputs = []
        lens = []

        for idx in indices:
            language, name = self.dataset_list[idx]
            targets.append( T.tensor( self.language_list.index(language), dtype=T.int64 ) )
            inputs.append( self._word_2_idx_tensor(name) )
            lens.append( len(name) )

        targets = T.tensor(targets, dtype=T.int64 )
        padded = pad_sequence(inputs, batch_first=True)
        lens = T.tensor(lens, dtype=T.int64 )
        return padded, targets, lens




class NameRNN(nn.Module):
    """ A very simple RNN with:
        - embedding layer, a GRU cell, linear out
    """
    def __init__(self, n_pos_chars, n_hidden, n_outputs, n_layers=1):
        super(NameRNN, self).__init__()

        ## Saving the network attributes
        self.n_pos_chars  = n_pos_chars
        self.n_hidden  = n_hidden # For both the embed vec and GRU
        self.n_outputs = n_outputs
        self.n_layers  = n_layers

        ## The layer stucture of the network
        self.embed = nn.Embedding(n_pos_chars, n_hidden)
        self.gru   = nn.GRU( n_hidden, n_hidden, n_layers, batch_first=True  )
        self.fc    = nn.Linear( n_hidden, n_outputs )

    def forward(self, input, lens=None):
        ## This is run over the entire batch of sequence (list of ints)
        ## Input_dims = Batch x Seq

        ## First we pass the list of ints through the embedding layer
        ## Result is now Batch x Seq x Vec(inpt/hid)
        embedded = self.embed( input )

        ## Pass through a packing function to increase comp eff
        if lens is not None:
            embedded = pack_padded_sequence( embedded, lens,
                                    batch_first=True, enforce_sorted=False )

        ## Pass through the GRU cell with new hidden layer
        hidden = self._reset_hidden( input.shape[0] )
        output, hidden = self.gru( embedded, hidden )

        ## Get the padded output form
        if lens is not None:
            output, output_lens = pad_packed_sequence( output, batch_first=True )

            ## Now we collect only the final outputs per sequence
            batch_idxes = [ i for i in range( len(input) ) ]
            fnl_outs = output[batch_idxes,output_lens-1]
        else:
            fnl_outs = hidden.squeeze(0)

        ## We use the last output of the GRU (stored in variable hidden)
        fc_out = self.fc(fnl_outs)

        return fc_out


    def _reset_hidden(self, batch_size):
        return T.zeros(self.n_layers, batch_size, self.n_hidden)


def main():

    ## We load the training set
    dataset = NameDataset( "../data/names/*.txt" )

    ## Initialise the RNN
    rnn = NameRNN( dataset.n_letters, 20, dataset.n_languages )

    ## Initialise the learning systems
    lr = 1e-3
    batch_size = 256
    opt = optim.Adam( rnn.parameters(), lr = lr )
    loss_fn = nn.CrossEntropyLoss()

    ## Now we begin the training loop
    for i in range(10000):
        opt.zero_grad()
        padded, targets, lens = dataset._get_rnd_batch( batch_size )
        output = rnn(padded, lens)
        loss = loss_fn( output, targets )
        loss.backward()
        opt.step()
        print(i, loss.item())
        if i%100==0:
            T.save(rnn.state_dict(), "saved_rnn")




if __name__ == '__main__':
    main()
