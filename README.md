# AIvisionPython
A school project, expanded upon. This neural network can recognize two distinct patterns but is highly generalized. Applying it for other purposes should be straightforward. This is an implementation of a neural network built on matrices and vectors.

### Network Structure
- **Number of layers**: 
  - The shape of the network is defined by an array passed to the constructor.
  - The length of the array represents the number of layers: `[input layer, ...hidden layers..., output layer]`.
  - The value at each index indicates the number of neurons in that layer.
- **Activation Functions**:
  - **Sigmoid**
  - **ReLU**: Used in hidden layers.
  - **Softmax**: Used in the output layer.
- **Training**:
  - Method: Stochastic Gradient Descent (SGD).
  - Loss Function: Mean Squared Error (MSE).

### Finetuned Configuration
- **Network Shape**: `[9, 39, 2]`.
- **Training Details**:
  - Capped at **1000 epochs** (average epochs needed: 26.841).
  - Learning rate: **0.1**.
  - Early stopping if cost falls below **0.01**.

## Notes
Activation functions can easily be swapped to all sigmoid if desired. However, experiments showed the default configuration to be optimal.

### References
The implementation is inspired by and built upon concepts from the following resources:
- [3Blue1Brown - Neural Networks](https://www.3blue1brown.com/topics/neural-networks) (Highly recommended!)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) (All chapters, highly recommended!)
- [Softmax Function and Its Derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
- [Convolutions and Backpropagation](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)
- *AI: A Modern Approach* by Stuart Russell and Peter Norvig (Sections 18.6 & 18.7)
- [MNIST NN from Scratch](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook)
- [Epoch vs Batch vs Mini-Batch](https://www.baeldung.com/cs/epoch-vs-batch-vs-mini-batch#:~:text=The%20mini%2Dbatch%20is%20a,of%20the%20dataset%20are%20used)

