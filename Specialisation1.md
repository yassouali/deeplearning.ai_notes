# Specialisation 1: Neural networks for Deep Learning

<!-- TOC -->

- [Specialisation 1: Neural networks for Deep Learning](#specialisation-1-neural-networks-for-deep-learning)
	- [Logistic Regression](#logistic-regression)
		- [Why we use Max Likelihood for classification ?](#why-we-use-max-likelihood-for-classification)
		- [Derivatives](#derivatives)
		- [Logitic regression as NN](#logitic-regression-as-nn)
	- [Notes <a name="notes"></a>](#notes-a-name%22notes%22a)
		- [Broadcasting & Numpy](#broadcasting--numpy)
		- [Activation function](#activation-function)
		- [Zero initiallisation](#zero-initiallisation)
	- [Shallow neural nets <a name="shallow_nets"></a>](#shallow-neural-nets-a-name%22shallownets%22a)
	- [Deep neural nets <a name="deep_nets"></a>](#deep-neural-nets-a-name%22deepnets%22a)
		- [Notations :](#notations)
		- [Forward and backprop](#forward-and-backprop)
			- [Forward propagation](#forward-propagation)
			- [Backpropagation](#backpropagation)
			- [Derivation of the backprop equations](#derivation-of-the-backprop-equations)

<!-- /TOC -->

## Logistic Regression
Parameters & inputs :
* X : inputs x batch
* W : inputs x outputs
* b: 1 x outputs

Logistic regression: Given X, we want <img src="https://latex.codecogs.com/gif.latex?$y_{predicted}&space;=&space;P(y=1&space;|&space;x)$" title="$y_{predicted} = P(y=1 | x)$" />

To have the output in the range of [0,1], to be interpreted as probability, we use a non linear activation function : sigmoid, to squash the results:

<img src="https://latex.codecogs.com/gif.latex?$$y_{predicted}&space;=&space;sigmoid(W^T&space;X&space;&plus;&space;b)\&space;=&space;\frac{1}{1&space;&plus;&space;e^{-(W^T&space;x&space;&plus;&space;b)}}&space;$$" title="$$y_{predicted} = sigmoid(W^T X + b)\ = \frac{1}{1 + e^{-(W^T x + b)}} $$" />

W : [Inputs x Outputs] and X : [Inputs x Batch]

Cost Function : Given Inputs and their Targets, our goal is to have the predictions to be equal to the Targets.
We define a loss function to be a convex function to be able to find its minimum using gradient descent.

### Why we use Max Likelihood for classification ?

When we perform binary classification (with classes 1 and -1),  we use the input X to arrive at a prediction <img src="https://latex.codecogs.com/gif.latex?$y_{predicted}$" title="$y_{predicted}$" /> where for <img src="https://latex.codecogs.com/gif.latex?$y_{predicted}$" title="$y_{predicted}$" />>=0, depending on the value of the output we classify that observation as class 1 or class -1.

How do we judge the performance? ideally, we want to maximize the accuracy, by giving a zero loss if sign of <img src="https://latex.codecogs.com/gif.latex?$y_{predicted}$" title="$y_{predicted}$" /> agree with the sign of the true class, i.e. <img src="https://latex.codecogs.com/gif.latex?$y.&space;y_{predicted}$" title="$y. y_{predicted}$" />>=0 . Otherwise, if we make the wrong prediction, the loss is 1.

Given that:
- If y = 1, <img src="https://latex.codecogs.com/gif.latex?$P(y|x)&space;=&space;y_{predicted}$" title="$P(y|x) = y_{predicted}$" />.
- If y = 0, <img src="https://latex.codecogs.com/gif.latex?$P(y|x)&space;=&space;1&space;-&space;y_{predicted}$" title="$P(y|x) = 1 - y_{predicted}$" />.

The loss function, that is the closer to such objective, while being a convex function is :

<img src="https://latex.codecogs.com/gif.latex?$$l(y,&space;y_{predicted})&space;=&space;-&space;(&space;y\&space;log(y_{predicted})&space;&plus;&space;(1&space;-&space;y)\&space;log(1&space;-&space;y_{predicted}))$$" title="$$l(y, y_{predicted}) = - ( y\ log(y_{predicted}) + (1 - y)\ log(1 - y_{predicted}))$$" />

If y = 0, so we want <img src="https://latex.codecogs.com/gif.latex?$y_{predicted}$" title="$y_{predicted}$" /> to be small, and this is done by minimising the loss <img src="https://latex.codecogs.com/gif.latex?$l&space;=&space;-&space;log(1&space;-&space;y_{predicted})$" title="$l = - log(1 - y_{predicted})$" />

Cost function measure the performances over all the training set, just the average loss over all example.

<img src="https://latex.codecogs.com/gif.latex?$$&space;J&space;=&space;\frac{1}{m}&space;\sum_{i=1}^m&space;\mathcal{L}(y^i,&space;y_{predicted}^i)$$" title="$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(y^i, y_{predicted}^i)$$" />

So we can write the probability as : <img src="https://latex.codecogs.com/gif.latex?$P(y|x)&space;=&space;y_{predicted}^y&space;\times&space;(1-&space;y_{predicted})^{(1&space;-&space;y)}$" title="$P(y|x) = y_{predicted}^y \times (1- y_{predicted})^{(1 - y)}$" />

The objective is to maximise P(y|x), given that for multiple example (iid), we'd like to maximise the product of their probabilities, so it is easier to use the log, which is a monotonic function, and maximise their summation, or minimise - loss.

### Derivatives

<img src="https://latex.codecogs.com/gif.latex?$$\frac{dl}{dw}&space;=&space;\frac{dl}{dy_{predicted}}&space;\frac{dy_{predicted}}{dz}&space;\frac{dz}{dw}$$\\&space;$$z&space;=&space;W^T&space;X&space;&plus;&space;b,\&space;\text{Thus}\&space;\frac{dz}{dw}&space;=&space;X$$\\&space;$$y_{predicted}&space;=&space;sigmoid(z),\&space;\text{Thus}\&space;\frac{dy_{predicted}}{dz}&space;=&space;y_{predicted}(1&space;-&space;y_{predicted})$$\\&space;$$\frac{dl}{dy_{predicted}}&space;=&space;\frac{(y_{predicted}&space;-&space;y)}{y_{predicted}(1&space;-&space;y_{predicted})}$$\\&space;$$\text{So}&space;:&space;\frac{dl}{dw}&space;=&space;X&space;(y_{predicted}&space;-&space;y)$$" title="$$\frac{dl}{dw} = \frac{dl}{dy_{predicted}} \frac{dy_{predicted}}{dz} \frac{dz}{dw}$$\\ $$z = W^T X + b,\ \text{Thus}\ \frac{dz}{dw} = X$$\\ $$y_{predicted} = sigmoid(z),\ \text{Thus}\ \frac{dy_{predicted}}{dz} = y_{predicted}(1 - y_{predicted})$$\\ $$\frac{dl}{dy_{predicted}} = \frac{(y_{predicted} - y)}{y_{predicted}(1 - y_{predicted})}$$\\ $$\text{So} : \frac{dl}{dw} = X (y_{predicted} - y)$$" />

### Logitic regression as NN

Logistic regression can also be viewed as a one layer neural net:

For one example <img src="https://latex.codecogs.com/gif.latex?$x^{(i)}$" title="$x^{(i)}$" />:

<img src="https://latex.codecogs.com/gif.latex?$$z^{(i)}&space;=&space;w^T&space;x^{(i)}&space;&plus;&space;b$$\\&space;\\&space;$$y_{predicted}^{(i)}&space;=&space;a^{(i)}&space;=&space;sigmoid(z^{(i)})$$\\&space;\\&space;$$&space;\mathcal{L}(a^{(i)},&space;y^{(i)})&space;=&space;-&space;y^{(i)}&space;\log(a^{(i)})&space;-&space;(1-y^{(i)}&space;)&space;\log(1-a^{(i)})$$" title="$$z^{(i)} = w^T x^{(i)} + b$$\\ \\ $$y_{predicted}^{(i)} = a^{(i)} = sigmoid(z^{(i)})$$\\ \\ $$ \mathcal{L}(a^{(i)}, y^{(i)}) = - y^{(i)} \log(a^{(i)}) - (1-y^{(i)} ) \log(1-a^{(i)})$$" />

Forward Propagation:
- Get an training set X (m examples)
- Compute <img src="https://latex.codecogs.com/gif.latex?$A&space;=&space;\sigma(w^T&space;X&space;&plus;&space;b)&space;=&space;(a^{(1)},&space;a^{(2)},&space;...,&space;a^{(m-1)},&space;a^{(m)})$" title="$A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$" />
- Calculate the cost function: <img src="https://latex.codecogs.com/gif.latex?J&space;=&space;-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})&plus;(1-y^{(i)})\log(1-a^{(i)})" title="J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})" />

Backward Propagation:
- Compute garadients with the cost respect to the weights <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J}{\partial&space;w}&space;=&space;\frac{1}{m}X(A-Y)^T" title="\frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T" />,
- And to the bias <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J}{\partial&space;b}&space;=&space;\frac{1}{m}&space;\sum_{i=1}^m&space;(a^{(i)}-y^{(i)})" title="\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})" />
- For a parameter <img src="https://latex.codecogs.com/gif.latex?$\theta$" title="$\theta$" /> (weights & bias), we update by <img src="https://latex.codecogs.com/gif.latex?$\theta&space;=&space;\theta&space;-&space;\alpha&space;\text{&space;}&space;d\theta$,&space;\text{where&space;}&space;$\alpha$" title="$\theta = \theta - \alpha \text{ } d\theta$, \text{where } $\alpha$" /> is the learning rate.

In order for Gradient Descent to work we must choose the learning rate wisely. The learning rate α determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

An example of logistic regression applied to cats:

<p align="center"><img src="figures/logistic_regression_cats.png" align="center" width="550"></p>

## Notes <a name="notes"></a>

### Broadcasting & Numpy

For a matrix of size (m, n), if we apply an operation using an (1, n) matrix, the latter will be broadcasted into (m, n), i.e., the row will be copied m times, same for an (m, 1) matrix, the column will be braodcasted n times into (m, n).

In we create a vector of random gussian values : `a = np.random.rand(5)`, this in case, a is rank one matrix with a shape of `(5, )`, if we do `a.T` we'll get the same output, and `a @ a.T` gives us a scalar instead of 5 x 5 matrix, it is easier to specify the dimensions of the desired matrix, such as `a = np.random.rand(5, 1)`

### Activation function
The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. But for binary classifiers, it is better to use singmoid functions as the outputs layer in case the output is in the range of [0,1].

### Zero initiallisation
Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons. Logistic regression's weights `w` should be initialized randomly rather than to all zeros, because if we initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to "break symmetry".

## Shallow neural nets <a name="shallow_nets"></a>
Logistic regression is very limited, we go directly from the inputs to the prediction, one possible way to improve the performance is to add a hidden layer to learn higher level abstraction of the inputs:

<p align="center"><img src="figures/shallow_net.png" align="center" width="450"></p>

For one example <img src="https://latex.codecogs.com/gif.latex?$x^{(i)}$" title="$x^{(i)}$" />:

<img src="https://latex.codecogs.com/gif.latex?$$z^{[1]&space;(i)}&space;=&space;W^{[1]}&space;x^{(i)}&space;&plus;&space;b^{[1]}$$\\&space;\\&space;$$a^{[1]&space;(i)}&space;=&space;\tanh(z^{[1]&space;(i)})$$\\&space;\\&space;$$z^{[2]&space;(i)}&space;=&space;W^{[2]}&space;a^{[1]&space;(i)}&space;&plus;&space;b^{[2]}$$\\&space;\\&space;$$y^{(i)}_{prediction}&space;=&space;a^{[2]&space;(i)}&space;=&space;\sigma(z^{&space;[2]&space;(i)})$$\\&space;\\&space;$$y^{(i)}_{prediction}&space;=&space;\begin{cases}&space;1&space;&&space;\text{if&space;}&space;a^{[2](i)}&space;>&space;0.5&space;\\&space;0&space;&&space;\text{otherwise&space;}&space;\end{cases}$$" title="$$z^{[1] (i)} = W^{[1]} x^{(i)} + b^{[1]}$$\\ \\ $$a^{[1] (i)} = \tanh(z^{[1] (i)})$$\\ \\ $$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2]}$$\\ \\ $$y^{(i)}_{prediction} = a^{[2] (i)} = \sigma(z^{ [2] (i)})$$\\ \\ $$y^{(i)}_{prediction} = \begin{cases} 1 & \text{if } a^{[2](i)} > 0.5 \\ 0 & \text{otherwise } \end{cases}$$" />

Given the predictions on all the examples, we can also compute the cost $J$ as follows :

<img src="https://latex.codecogs.com/gif.latex?$$J&space;=&space;-&space;\frac{1}{m}&space;\sum\limits_{i&space;=&space;0}^{m}&space;\large\left(\small&space;y^{(i)}\log\left(a^{[2]&space;(i)}\right)&space;&plus;&space;(1-y^{(i)})\log\left(1-&space;a^{[2]&space;(i)}\right)&space;\large&space;\right)&space;\small$$" title="$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large \right) \small$$" />

The general methodology to build a Neural Network is to:
1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
2. Initialize the model's parameters
3. Loop:
	- Implement forward propagation
	- Compute loss
	- Implement backward propagation to get the gradients
	- Update parameters (gradient descent)

## Deep neural nets <a name="deep_nets"></a>
Now instead of having only one hidden layer, we can add as many as we want to give the network the ability to computes more complex functions, and the same computations we've seen above are applied:

### Notations :

- Superscript [l] denotes a quantity associated with the <img src="https://latex.codecogs.com/gif.latex?$l^{th}$" title="$l^{th}$" /> layer.
    - Example: <img src="https://latex.codecogs.com/gif.latex?$a^{[L]}$" title="$a^{[L]}$" /> is the <img src="https://latex.codecogs.com/gif.latex?$L^{th}$" title="$L^{th}$" /> layer activation. <img src="https://latex.codecogs.com/gif.latex?$W^{[L]}$" title="$W^{[L]}$" /> and <img src="https://latex.codecogs.com/gif.latex?$b^{[L]}$" title="$b^{[L]}$" /> are the <img src="https://latex.codecogs.com/gif.latex?$L^{th}$" title="$L^{th}$" /> layer parameters.
- Superscript (i) denotes a quantity associated with the <img src="https://latex.codecogs.com/gif.latex?$i^{th}$" title="$i^{th}$" /> example.
    - Example: <img src="https://latex.codecogs.com/gif.latex?$x^{(i)}$" title="$x^{(i)}$" /> is the <img src="https://latex.codecogs.com/gif.latex?$i^{th}$" title="$i^{th}$" /> training example.
- Lowerscript i denotes the <img src="https://latex.codecogs.com/gif.latex?$i^{th}$" title="$i^{th}$" /> entry of a vector.
    - Example: <img src="https://latex.codecogs.com/gif.latex?$a^{[l]}_i$" title="$a^{[l]}_i$" /> denotes the <img src="https://latex.codecogs.com/gif.latex?$i^{th}$" title="$i^{th}$" /> entry of the <img src="https://latex.codecogs.com/gif.latex?$l^{th}$" title="$l^{th}$" /> layer's activations)

<p align="center"><img src="figures/deep_net.png" align="center" width="600"></p>

### Forward and backprop
<p align="center"><img src="figures/forward_and_backprop.png" align="center" width="600"></p>

#### Forward propagation

In forward propagation, we apply the same operation <img src="https://latex.codecogs.com/gif.latex?$Z^{[l]}&space;=&space;W^{[l]}A^{[l-1]}&space;&plus;b^{[l]}$" title="$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$" /> to go from one layer (l - 1) to the next (l) using the weight matrix <img src="https://latex.codecogs.com/gif.latex?$W^{[l]}$" title="$W^{[l]}$" /> and bias <img src="https://latex.codecogs.com/gif.latex?$b^{[l]}$" title="$b^{[l]}$" />.
- In general the operations applied are: (LINEAR -> ACTIVATION) x (L-1) -> LINEAR -> SIGMOID
	- LINEAR : <img src="https://latex.codecogs.com/gif.latex?$Z^{[l]}&space;=&space;W^{[l]}A^{[l-1]}&space;&plus;b^{[l]}$" title="$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$" />
	- ACTIVATION : either<img src="https://latex.codecogs.com/gif.latex?$A^{[l-1]}&space;=&space;Relu(Z^{[l]})&space;\text{&space;or&space;}&space;A^{[l-1]}&space;=&space;\tanh(Z^{[l]})$" title="$A^{[l-1]} = Relu(Z^{[l]}) \text{ or } A^{[l-1]} = \tanh(Z^{[l]})$" />
	- SIGMOID : <img src="https://latex.codecogs.com/gif.latex?$a^{[L]}&space;=&space;y_{predicted}&space;=&space;\sigma(Z^{[L]})$" title="$a^{[L]} = y_{predicted} = \sigma(Z^{[L]})$" />
- After that we compute the (binary) cross-entropy cost J, using the following formula: <img src="https://latex.codecogs.com/gif.latex?-\frac{1}{m}&space;\sum\limits_{i&space;=&space;1}^{m}&space;(y^{(i)}\log\left(a^{[L]&space;(i)}\right)&space;&plus;&space;(1-y^{(i)})\log\left(1-&space;a^{[L](i)}\right))" title="-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right))" />

#### Backpropagation
In backpropagation, the goal is to calculate the derivatives of the cost function with respect to the parameters of the model (the weights and biases), to be able to update them using gradient descent.

The backpropagation is very similar to the forward propagation, the difference is that this time instead of calculating the activations for the first layer to the last, we'll backpropagate the error from the last layer to the first, each time step, by calculating the derivatives of the cost with respect to the activations <img src="https://latex.codecogs.com/gif.latex?\frac{\partial{J}}{\partial{a^l}}" title="\frac{\partial{J}}{\partial{a^l}}" />. and reuse <img src="https://latex.codecogs.com/gif.latex?\frac{\partial{J}}{\partial{a^l}}" title="\frac{\partial{J}}{\partial{a^l}}" /> of the last layer to calculate the derivatives of the cost with respect to the weights and biases.

<p align="center"><img src="figures/back_prop.png" align="center" width="700"></p>

#### Derivation of the backprop equations

For layer L:

<img src="https://latex.codecogs.com/gif.latex?$$\frac{\partial{L}}{\partial{w_{hj}^L}}&space;=&space;\frac{\partial{L}}{\partial{a_{h}^L}}&space;\frac{\partial{a_{h}^L}}{\partial{z_{h}^L}}&space;\frac{\partial{z_{h}^L}}{\partial{w_{hj}^L}}&space;$$&space;\\&space;\\&space;$$\frac{\partial{L}}{\partial{a_{h}^L}}&space;=&space;(\hat{y_h}&space;-&space;y)$$&space;\\&space;\\&space;$$&space;\frac{\partial{a_{h}^L}}{\partial{z_{h}^L}}&space;=&space;\sigma(a_{h}^L)&space;(1-&space;\sigma(a_{h}^L))$$&space;\\&space;\\&space;$$&space;\frac{\partial{z_{h}^L}}{\partial{w_{hj}^L}}&space;=&space;a_j&space;$$" title="$$\frac{\partial{L}}{\partial{w_{hj}^L}} = \frac{\partial{L}}{\partial{a_{h}^L}} \frac{\partial{a_{h}^L}}{\partial{z_{h}^L}} \frac{\partial{z_{h}^L}}{\partial{w_{hj}^L}} $$ \\ \\ $$\frac{\partial{L}}{\partial{a_{h}^L}} = (\hat{y_h} - y)$$ \\ \\ $$ \frac{\partial{a_{h}^L}}{\partial{z_{h}^L}} = \sigma(a_{h}^L) (1- \sigma(a_{h}^L))$$ \\ \\ $$ \frac{\partial{z_{h}^L}}{\partial{w_{hj}^L}} = a_j $$" />

So first we calculate the error / loss, and then its derivative with respect to the ouputs (here we use L2 loss), pass thought the non linearity, and then to get the derivatives with respect to the parameters, we multiply by the inputs of the previous layer $a_j$

The next step is to back propagate the derivatives of the cost with respect to the activation of layer L to layer L-1 and get the derivatives with respect to the activations of layer L-1, and then we get derivatives with respect to the parameters the same way, by multiplying the local derivatives to the inputs of the previous layer L-2:

For layer L - 1:

<img src="https://latex.codecogs.com/gif.latex?$$&space;\frac{\partial{L}}{\partial{w_{jk}^{L-1}}}&space;=&space;\frac{\partial{L}}{\partial{a_{k}^{L-1}}}&space;\frac{a_{k}^{L-1}}{\partial{z_{k}^{L-1}}}&space;\frac{\partial{z_{k}^{L-1}}}{\partial{w_{jk}^{L-1}}}&space;$$" title="$$ \frac{\partial{L}}{\partial{w_{jk}^{L-1}}} = \frac{\partial{L}}{\partial{a_{k}^{L-1}}} \frac{a_{k}^{L-1}}{\partial{z_{k}^{L-1}}} \frac{\partial{z_{k}^{L-1}}}{\partial{w_{jk}^{L-1}}} $$" />

We get the values with the same equations, the only difference this time, is to get <img src="https://latex.codecogs.com/gif.latex?\frac{\partial{L}}{\partial{a_{k}^{L-1}}}" title="\frac{\partial{L}}{\partial{a_{k}^{L-1}}}" /> we need to sum over the upcoming gradients from the last layer:

<img src="https://latex.codecogs.com/gif.latex?$$&space;\frac{\partial{L}}{\partial{a_{k}^{L-1}}}&space;=&space;\sum_h&space;\frac{\partial{z_{h}^{L}}}{\partial{a_{k}^{L-1}}}&space;\frac{\partial{a_{h}^{L}}}{\partial{z_{h}^{L}}}&space;\frac{\partial{L}}{\partial{a_{h}^{L}}}&space;=&space;\sum_h&space;W_{hj}&space;\frac{\partial{a_{h}^{L}}}{\partial{z_{h}^{L}}}&space;\frac{\partial{L}}{\partial{a_{h}^{L}}}&space;$$" title="$$ \frac{\partial{L}}{\partial{a_{k}^{L-1}}} = \sum_h \frac{\partial{z_{h}^{L}}}{\partial{a_{k}^{L-1}}} \frac{\partial{a_{h}^{L}}}{\partial{z_{h}^{L}}} \frac{\partial{L}}{\partial{a_{h}^{L}}} = \sum_h W_{hj} \frac{\partial{a_{h}^{L}}}{\partial{z_{h}^{L}}} \frac{\partial{L}}{\partial{a_{h}^{L}}} $$" />

This is done until we get the input layer, where we end up with the derivatives of the cost with respect the inputs (we don't use them generally, but can be used for some application like style transfer when we update the image and not the parameters of the model).