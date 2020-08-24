import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

def word_2_idx( word, char_list ):
    return [ char_list.index(char) for char in word ]

def idx_2_word( idxes, char_list ):
    return ''.join([ char_list[idx] for idx in idxes ])

def main():

    ## Define our encoding and our input/output
    char_list = [ 'h', 'i', 'e', 'l', 'o' ]
    in_word  = "hihell"
    out_word = "ihello"


    ## Convert our input/output to idx
    in_data  = word_2_idx( in_word,  char_list )
    out_data = word_2_idx( out_word, char_list )

    ## Convert our input idx to onehot using an identity lookup
    id_lookup = np.identity(len(char_list))
    in_hot = [ id_lookup[x] for x in in_data ]
    input  = T.tensor([in_hot], dtype=T.float32)

    ## Alternatively we embed using a trainable lookup
    emb = nn.Embedding( len(char_list), 5 )
    in_emb = emb(T.tensor(in_data))
    input = in_emb.unsqueeze(0).detach()

    target = T.tensor(out_data, dtype=T.int64)

    ## Parameters of network and data
    n_in = 5
    n_out = 5

    ## Define the RNN Cell
    cell = nn.RNN( input_size = n_in, hidden_size = n_out, batch_first = True )
    opt = optim.Adam( cell.parameters(), lr=1e-3 )
    loss_fn = nn.CrossEntropyLoss()

    ## Pass the input through the network
    for i in range(1000):
        opt.zero_grad()
        hidden = T.zeros( 1, 1, n_out )
        out, hidden = cell( input, hidden )
        loss = loss_fn( out.squeeze(), target )
        loss.backward()
        opt.step()

        max_idxes = out.squeeze().detach().argmax(1)
        word_out = idx_2_word(max_idxes, char_list)
        print(word_out)


if __name__ == '__main__':
    main()
