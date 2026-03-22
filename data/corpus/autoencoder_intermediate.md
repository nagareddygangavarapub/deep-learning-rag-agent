# Autoencoders

## What is an Autoencoder?
An autoencoder is a type of neural network trained to reconstruct its
input at the output layer. It consists of two parts: an encoder that
compresses the input into a lower-dimensional latent representation,
and a decoder that reconstructs the original input from that
representation. Introduced by Hinton and Salakhutdinov in 2006,
autoencoders learn efficient data representations in an unsupervised
manner without requiring labeled data.

## The Encoder
The encoder maps the input to a compressed latent space representation,
also called the bottleneck or code. It consists of one or more layers
that progressively reduce the dimensionality of the input. For example,
an encoder might compress a 784-dimensional image (28x28 pixels) down
to a 32-dimensional latent vector. The encoder learns which features
are most important to preserve for reconstruction.

## The Bottleneck
The bottleneck is the compressed representation at the center of the
autoencoder. Its dimensionality is smaller than the input, forcing the
network to learn a compact representation that captures the most important
features. The size of the bottleneck is a key hyperparameter — too small
and the network cannot capture enough information, too large and it may
simply copy the input without learning meaningful features.

## The Decoder
The decoder reconstructs the original input from the bottleneck
representation. It mirrors the encoder architecture, progressively
increasing dimensionality until it matches the input size. The decoder
learns to map points in latent space back to the original data space.
The reconstruction loss — typically mean squared error for continuous
inputs — measures how well the decoder recovers the original input.

## Variational Autoencoders
A Variational Autoencoder (VAE) extends the standard autoencoder by
learning a probability distribution over the latent space rather than
a fixed encoding. The encoder outputs a mean and variance, and the
latent code is sampled from this distribution. This forces the latent
space to be continuous and well-structured, enabling generation of new
samples by sampling from the prior distribution. VAEs are generative
models used for image generation and data augmentation.

## Applications of Autoencoders
Autoencoders are used for dimensionality reduction as an alternative to
PCA, anomaly detection by measuring reconstruction error on unseen data,
image denoising by training on noisy inputs with clean targets, and
feature learning for downstream classification tasks. The learned latent
representations capture the underlying structure of the data in a
compressed form that can be used by other models.
