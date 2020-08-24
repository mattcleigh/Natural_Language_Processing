import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from itertools import count

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from data import *
from network import *

def get_random_example(language_list, class_dict):

    ## Choose a random language and sample a name
    lang = rd.choice(language_list)
    name = rd.choice(class_dict[lang])

    ## Convert to tensors for training
    lang_tens = T.tensor([language_list.index(lang)], dtype=T.long)
    name_tens = word_to_tensor(name)

    return lang, name, lang_tens, name_tens


def train( network, opt, loss_fn, lang_tens, name_tens ):
    hidden = network.reset_hidden()
    opt.zero_grad()

    ## Push the sequence through the RNN, one letter at a time
    for i in range(len(name_tens)):
        output, hidden = network( name_tens[i], hidden )
    loss = loss_fn(output, lang_tens)
    loss.backward()
    opt.step()
    return output, loss.item()

def main():

    ## We load the training set
    all_files = "../data/names/*.txt"
    language_list, class_dict = build_dataset(all_files)
    n_languages = len(language_list)

    ## We create the simple RNN
    lr = 1e-3
    n_hidden = 128
    plt_every = 1000

    loss_fn = nn.CrossEntropyLoss()
    rnn = SimpleRNN( n_letters, n_hidden, n_languages )
    optimisor = optim.Adam( rnn.parameters(), lr = lr )

    ## Training the network
    avg_loss = 0
    loss_hist = []
    plt.ion()
    fig = plt.figure( figsize = (5,5) )
    ax = fig.add_subplot(111)
    line, = ax.plot( loss_hist, "x-" )

    for i in count(1):
        _, _, lang_tens, name_tens = get_random_example( language_list, class_dict )
        _, loss = train( rnn, optimisor, loss_fn, lang_tens, name_tens )
        avg_loss += loss
        if i%plt_every==0:
            avg_loss /= plt_every
            loss_hist.append(avg_loss)
            line.set_data( np.arange(len(loss_hist)), loss_hist )
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            avg_loss = 0
            T.save(rnn.state_dict(), "saved_rnn")



if __name__ == "__main__":
    main()
