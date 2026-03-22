# Recurrent Neural Networks (RNN) and LSTM

## What is an RNN?
A Recurrent Neural Network (RNN) is a type of neural network designed for
sequential data such as text, speech, and time series. Unlike feedforward
networks, RNNs have connections that form directed cycles, allowing them to
maintain a hidden state that acts as memory. At each time step, the hidden
state is updated based on the current input and the previous hidden state.
RNNs were introduced by Elman in 1990.

## The Vanishing Gradient Problem
The vanishing gradient problem occurs when gradients become exponentially
small as they are backpropagated through many time steps. This makes it
difficult for standard RNNs to learn long-range dependencies — the network
effectively forgets information from many steps ago. The exploding gradient
problem is the opposite, where gradients grow exponentially, causing unstable
training. Gradient clipping is used to address exploding gradients.

## Long Short-Term Memory (LSTM)
Long Short-Term Memory networks (LSTMs) were introduced by Hochreiter and
Schmidhuber in 1997 to solve the vanishing gradient problem. LSTMs introduce
a cell state — a separate memory pathway that runs through the entire
sequence. The cell state is modified by three learned gates: the forget gate,
the input gate, and the output gate. This gating mechanism allows LSTMs to
selectively remember or forget information over long sequences.

## The Forget Gate
The forget gate decides what information to discard from the cell state. It
takes the previous hidden state and current input, passes them through a
sigmoid function, and outputs values between 0 and 1. A value close to 0
means forget this information completely; a value close to 1 means keep it.
This allows the LSTM to reset its memory when it encounters a new context,
such as a new sentence in a document.

## The Input Gate
The input gate decides what new information to add to the cell state. It
consists of two parts: a sigmoid layer that decides which values to update,
and a tanh layer that creates a vector of new candidate values. The product
of these two determines what gets added to the cell state. This allows the
LSTM to selectively incorporate new information while ignoring irrelevant
inputs.

## The Output Gate
The output gate determines what part of the cell state to expose as the
hidden state output. It applies a sigmoid to the input and hidden state to
decide which parts of the cell state to output, then multiplies by a tanh
of the cell state. The resulting hidden state is passed to the next time
step and used for predictions. This separation of cell state and hidden
state is the key innovation of the LSTM architecture.

## Seq2Seq Models
Sequence-to-sequence (Seq2Seq) models use an encoder-decoder architecture
built from RNNs or LSTMs. The encoder processes the input sequence and
compresses it into a fixed-length context vector. The decoder generates
the output sequence from this context vector. Seq2Seq was introduced by
Sutskever, Vinyals, and Le in 2014 and enabled neural machine translation.
The attention mechanism was later added to overcome the bottleneck of
fixed-length context vectors.
