import sys
import torch as T
import torch.nn.functional as F

from data import *
from network import *

rnn = SimpleRNN( 55, 128, 18 )
rnn.load_state_dict(T.load("saved_rnn"))

def process(word):
    hidden = rnn.reset_hidden()
    for i in range(len(word)):
        output, hidden = rnn(word[i], hidden)
    return output

def category_from_out(output, language_list):
    return( language_list[ T.argmax(output) ] )

def predict(name):

    ## We load the list of possible languages
    all_files = "../data/names/*.txt"
    language_list, _ = build_dataset(all_files)

    output = process(word_to_tensor(name))
    probs = T.exp(output)[0]
    probs = probs / T.max(probs) * 20
    sort = T.argsort(probs, descending=True)

    for s in sort:
        count = int(probs[s].item())
        if count > 0:
            print( "{:10}".format(language_list[s]), "#" * count )


if __name__ == '__main__':
    predict(sys.argv[1])
