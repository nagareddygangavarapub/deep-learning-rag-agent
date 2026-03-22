# Artificial Neural Networks (ANN)

## What is an Artificial Neural Network?
An Artificial Neural Network (ANN) is a computational model inspired by the
structure of biological neural networks in the human brain. It consists of
layers of interconnected nodes (neurons) that process information using
connectionist approaches. ANNs are the foundation of modern deep learning
and are used for tasks such as classification, regression, and pattern
recognition.

## Layers in a Neural Network
A typical ANN has three types of layers: the input layer, one or more hidden
layers, and the output layer. The input layer receives raw data. Hidden layers
perform intermediate computations by applying weights, biases, and activation
functions. The output layer produces the final prediction. The depth of a
network refers to the number of hidden layers — deeper networks can learn
more complex representations.

## Weights and Biases
Weights and biases are the learnable parameters of a neural network. Each
connection between neurons has an associated weight that determines its
strength. A bias term is added to each neuron to shift the activation
function. During training, weights and biases are updated using
backpropagation and gradient descent to minimize the loss function.

## Activation Functions
Activation functions introduce non-linearity into the network, allowing it
to learn complex patterns. Common activation functions include ReLU
(Rectified Linear Unit), sigmoid, and tanh. ReLU (f(x) = max(0, x)) is
the most widely used because it mitigates the vanishing gradient problem
and is computationally efficient. Sigmoid maps values to (0,1) and is
used in binary classification output layers.

## Backpropagation
Backpropagation is the algorithm used to train neural networks. It computes
the gradient of the loss function with respect to each weight using the
chain rule of calculus. The gradients flow backwards from the output layer
to the input layer. These gradients are then used by an optimizer such as
SGD or Adam to update the weights and reduce the loss. Backpropagation was
popularized by Rumelhart, Hinton, and Williams in 1986.

## Gradient Descent and Optimizers
Gradient descent is the optimization algorithm that updates weights in the
direction that minimizes the loss. Stochastic Gradient Descent (SGD) updates
weights after each training example. Mini-batch gradient descent updates
after a small batch. Adam (Adaptive Moment Estimation) combines momentum
and adaptive learning rates, making it the most popular optimizer for deep
learning because it converges faster and requires less tuning.

## Overfitting and Regularization
Overfitting occurs when a model performs well on training data but poorly
on unseen data. Regularization techniques prevent overfitting. L2
regularization (weight decay) penalizes large weights. Dropout randomly
deactivates neurons during training, forcing the network to learn redundant
representations. Early stopping halts training when validation loss stops
improving.
