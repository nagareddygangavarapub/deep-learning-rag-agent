# Convolutional Neural Networks (CNN)

## What is a CNN?
A Convolutional Neural Network (CNN) is a type of deep neural network
designed for processing structured grid data such as images. CNNs use
convolutional layers to automatically learn spatial hierarchies of features
— from edges and textures at lower layers to complex shapes and objects at
higher layers. They were popularized by LeCun et al. in the 1998 LeNet
architecture for handwritten digit recognition.

## Convolutional Layers
A convolutional layer applies a set of learnable filters (kernels) to the
input. Each filter slides across the input and computes a dot product,
producing a feature map. Filters detect local patterns such as edges,
corners, and textures. Using the same filter across the entire input is
called weight sharing — this dramatically reduces the number of parameters
compared to a fully connected layer and makes CNNs translation invariant.

## Pooling Layers
Pooling layers reduce the spatial dimensions of feature maps, decreasing
computation and providing translation invariance. Max pooling takes the
maximum value in each pooling window, preserving the most prominent features.
Average pooling takes the mean. Pooling makes the representation more
compact and helps the network focus on the presence of features rather than
their exact location.

## Receptive Field
The receptive field of a neuron is the region of the input that influences
its output. In deep CNNs, neurons in higher layers have larger receptive
fields and respond to more global features. Stacking multiple convolutional
layers increases the effective receptive field, allowing the network to
capture both local and global patterns hierarchically.

## Famous CNN Architectures
AlexNet (Krizhevsky et al., 2012) won the ImageNet competition and
demonstrated the power of deep CNNs with ReLU activations and dropout.
VGGNet used very small 3x3 filters stacked deeply. ResNet introduced
skip connections (residual connections) that allow gradients to flow
directly through the network, enabling training of very deep networks
(100+ layers) without vanishing gradients.

## CNNs vs Fully Connected Networks
CNNs are more efficient than fully connected networks for image data
because of weight sharing and local connectivity. A fully connected layer
connecting a 224x224 image to 1000 neurons would require 50 million
parameters. A convolutional layer with 64 filters of size 3x3 requires
only 1,728 parameters regardless of input size. This parameter efficiency
makes CNNs scalable to large images.
