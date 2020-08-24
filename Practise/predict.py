import sys
import torch as T
import torch.nn.functional as F

from Name_Classification import *

def category_from_out(output, language_list):
    return( language_list[ T.argmax(output) ] )

def predict(name):

    ## We load the training set
    dataset = NameDataset( "../data/names/*.txt" )

    ## Initialise the RNN
    rnn = NameRNN( dataset.n_letters, 20, dataset.n_languages )
    rnn.load_state_dict(T.load("saved_rnn"))

    word_tens = dataset._word_2_idx_tensor(name).unsqueeze(0)

    output = rnn( word_tens )
    probs = T.exp(output)[0]
    probs = probs / T.max(probs) * 20
    sort = T.argsort(probs, descending=True)

    for s in sort:
        count = int(probs[s].item())
        if count > 0:
            print( "{:10}".format(dataset.language_list[s]), "#" * count )


if __name__ == '__main__':
    predict(sys.argv[1])
