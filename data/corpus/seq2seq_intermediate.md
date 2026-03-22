# Sequence-to-Sequence Models (Seq2Seq)

## What is Seq2Seq?
Sequence-to-sequence (Seq2Seq) models are a class of neural network
architectures designed to transform one sequence into another. Introduced
by Sutskever, Vinyals, and Le in 2014, Seq2Seq models enabled neural
machine translation, converting sentences from one language to another.
The architecture consists of two components: an encoder that reads the
input sequence and a decoder that generates the output sequence.

## The Encoder
The encoder processes the input sequence one token at a time and compresses
the entire input into a fixed-length context vector, also called the thought
vector. The encoder is typically an RNN or LSTM that updates its hidden
state at each time step. The final hidden state of the encoder captures
a summary of the entire input sequence and is passed to the decoder as
the starting point for generation.

## The Decoder
The decoder generates the output sequence one token at a time, conditioned
on the context vector from the encoder. At each step, the decoder takes
its previous hidden state, the previously generated token, and the context
vector to produce the next token. Generation continues until a special
end-of-sequence token is produced. The decoder is also typically an RNN
or LSTM.

## The Bottleneck Problem
The fixed-length context vector is a bottleneck in the Seq2Seq architecture.
For long input sequences, compressing all information into a single vector
causes information loss. The encoder must discard details to fit everything
into the context vector, which degrades translation quality for long
sentences. This limitation motivated the development of the attention
mechanism, which allows the decoder to look back at all encoder hidden
states rather than just the final one.

## Attention Mechanism
The attention mechanism, introduced by Bahdanau et al. in 2015, allows
the decoder to selectively focus on different parts of the input sequence
at each decoding step. Instead of using only the final encoder hidden state,
attention computes a weighted sum of all encoder hidden states. The weights
are learned and indicate which input tokens are most relevant for generating
each output token. Attention dramatically improved translation quality and
became the foundation for the Transformer architecture.

## Applications of Seq2Seq
Seq2Seq models are used in machine translation, text summarization,
question answering, and dialogue systems. The encoder-decoder pattern
generalizes beyond text — it is used in image captioning (CNN encoder,
RNN decoder) and speech recognition (audio encoder, text decoder).
The architecture established the template that modern large language
models build upon.
