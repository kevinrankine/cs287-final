# cs287-final
Final Project for CS287: Question Retrival

Entrypoint is main.lua

models contained in models/

count-based in models/CountModel.lua

all others in models/NeuralEncoder.lua

models/FixedLookupTable.lua and models/MaxMarginCriterion are a fixed word embedding and a loss criterion implementation.

## args

-model : selects which model

-d_hid : size of rnn hidden state

-eta : learning rate

-nepochs : number of epochs of training

-margin : margin for the loss function

-nbatches : number of examples in each batch

-dropout : dropout value to use (0 if no dropout)

-pool : state aggregation to use [mean or last]

-kernel_width : kernel width for the CNN

-cuda : 1 if use GPU 0 o.w.

-from_file : File from which to load model

-to_file : File to save model to

-train 1 : if train the model 0 o.w.

-body : 1 if use body else 0


