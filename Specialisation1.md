# Specialisation 1: Neural networks for Deep Learning

## Logistic Regression

Parameters & inputs :
* X : inputs x batch
* W : inputs x outputs
* b: 1 x outputs

Logistic regression: Given X, we want y_predicted = P(y=1 | x)

To have the output in the range of [0,1], to be interpreted as probability, 

we use a non linear activation function : sigmoid, to squash the results:

			y_predicted = sigmoid(W.T @ X + b) / W : [Inputs x Outputs] and X : [Inputs x Batch]

Cost Function : Given Inputs and their Targets, our goal is to have the predictions to be equal to the Targets
We define a loss function to be a convex function, and be able to find its minimum, if we use a MSE loss function, the 
loss function becomes non convexe, more details :

#### Side not, why we use Max Likelihood for classification
When we perform binary classification model (with classes 1 and -1),  we use the input X to arrive at a continuous prediction y_predicted where for y_predicted>=0, we classify that observation as class 1, otherwise to class -1.

How do we judge the performance, ideally, we want to maximize the accuracy, by giving a zero loss if sign of y_predicted agree with the sign of the true class, i.e. y y_predicted>=0 . Otherwise, if we make the wrong prediction, the loss is 1.

The loss function, that is the closer to such objective, while being a convex function is 

		l(y, y_predicted) = - ( y log(y_predicted) + (1 - y) log(1 - y_predicted) )
		If y = 0, loss = - log(y_predicted), so we want it to be small

Cost function measure the performances over all the training set, just the average loss over all example.

		If y = 1, P(y|x) = y_predicted
		If y = 0, P(y|x) = 1 - y_predicted
So we can write the probability as : P(y|x) = y_predicted^y (1- y_predicted)^(1 - y)

The objective is to maximise P(y|x), given that for multiple example (iid), we'd like to maximise the product of their probabilities, so it is easier to use the lod, which is a monotonic function, and maximise their summation, or minimise - loss.


#### Derivatives

		dl/dw = dl/dy_predicted x dy_predicted/dz x dz/dw
		z = W.T @ X + b, so dz/dw = X
		y_predicted = sigmoid(z), dy_predicted/dz = a(1 - a)
		dl/dy_predicted = (y_predicted - y) / y_predicted(1 - y_predicted)
		So : dl/dw = X (y_predicted - y)

## Shallow neural nets

#### Broadcasting & Numpy

For a matrix of size (m, n), if we apply an operation using an (1, n) matrix, the latter will be broadcasted into (m, n), i.e., the row will be copied m times, same for an (m, 1) matrix, the column will be braodcasted n times into (m, n).

In we create a vector of random gussian values : `a = np.random.rand(5)`, this in case, a is rank one matrix with a shape of (5,), if we do a.T we'll get the same output, and a @ a.T gives us a scalar instead of 5 x 5 matrix, it is easier to specify the dimensions of the desired matrix, such as `a = np.random.rand(5, 1)`

#### Activation function
The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. But for binary classifiers, it is better to use singmoid functions in the outputs layer give that the output is in the range of [0,1].

#### Zero initiallisation
Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons. Logistic regression's weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to "break symmetry"

## Deep neural nets

