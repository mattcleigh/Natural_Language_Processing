import torch as T
import torch.nn as nn

def main():

    ## Define our encoding for "hello"
    h = [ 1, 0, 0, 0 ]
    e = [ 0, 1, 0, 0 ]
    l = [ 0, 0, 1, 0 ]
    o = [ 0, 0, 0, 1 ]

    word = [ h, e, l, l, o ]

    ## Features of network
    n_in = 4
    n_out = 2

    ## Define the RNN Cell
    cell = nn.GRU( input_size = n_in, hidden_size = n_out, batch_first = True )

    ## We can manually give sequence one at a time in a loop
    hidden = T.zeros( 1, 1, n_out ) # [nlyrs=1, batch=1, out=2]
    for letter in word:
        input = T.tensor([[letter]], dtype=T.float32) # [batch=1, seq=1, inp=4]
        out, hidden = cell( input, hidden )
    print(out, hidden)

    ## We can pass a whole sequence together, no for loop
    hidden = T.zeros( 1, 1, n_out ) # [nlyrs=1, batch=1, out=2]
    input = T.tensor([word], dtype=T.float32) # [batch=1, seq=5, inp=4]
    out, hidden = cell( input, hidden )
    print(out, hidden) # Output is whole seq, hidden only includes last

    ## Finally we can process this in batch form
    hidden = T.zeros( 1, 3, n_out ) # [nlyrs=1, batch=3, out=2]
    word1 = [ h, e, l, l, o ]
    word2 = [ o, e, o, l, l ]
    word3 = [ h, l, e, h, o ]
    batch = [ word1, word2, word3 ]
    input = T.tensor(batch, dtype=T.float32) # [batch=3, seq=5, inp=4]
    out, hidden = cell( input, hidden )
    print(out, hidden) # Output is whole seq, hidden only includes last



if __name__ == '__main__':
    main()
