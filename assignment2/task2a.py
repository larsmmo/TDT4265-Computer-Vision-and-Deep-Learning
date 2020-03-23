import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray, mean = 33.29597520727041, std = 78.54199339362128):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    bias = np.ones((X.shape[0],1))

    X = (X - mean) / std

    X = np.append(X, bias, axis = 1)

    assert X.shape[1] == 785,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    N = targets.shape[0]

    ce = -(1/(N)) * np.sum(np.sum(targets*np.log(outputs)))

    return ce


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Buffers for gradient calculation
        self.zj = [None for i in range(len(neurons_per_layer) - 1)]
        self.aj = [None for i in range(len(neurons_per_layer) - 1)]

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        # Initialize weights
        if use_improved_weight_init:
        	self.ws[0] = np.random.normal(0, 1/np.sqrt(self.I), self.ws[0].shape)
        	for i in range(len(self.neurons_per_layer) - 1):
        		self.ws[i+1] = np.random.normal(0, 1/np.sqrt(self.neurons_per_layer[i]), self.ws[i+1].shape)
        else:
        	self.ws[0] = np.random.uniform(-1, 1, (self.I, self.neurons_per_layer[0]))
        	for i in range(len(self.neurons_per_layer) - 1):
        		self.ws[i+1] = np.random.uniform(-1, 1, (self.neurons_per_layer[i], self.neurons_per_layer[i+1]))

    def forward(self, X: np.ndarray, layer = 0) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # First layer, sigmoid activation function
        self.zj[layer] = np.dot(X, self.ws[layer])

        if self.use_improved_sigmoid:
        	self.aj[layer] = 1.7159* np.tanh((2.0/3.0)*self.zj[layer])
        else:
        	self.aj[layer] = 1/(1 + np.exp(-self.zj[layer]))

        if (layer+1) < (len(self.neurons_per_layer) -1):		# For all hidden layers, update zj and aj
        	layer = layer + 1
        	out = self.forward(self.aj[layer-1], layer)	
        	return out											

        # To output layer, softmax function
        zk = np.dot(self.aj[-1], (self.ws[-1]))
        ez = np.exp(zk)
        denominator = np.tile(sum(ez.T), (self.ws[-1].shape[1], 1))

        y = ez / denominator.T

        return y

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
       	delta = -(targets - outputs)
       	self.grads[-1] = np.dot(delta.T, self.aj[-1]).T / (outputs.shape[0])

       	for i in range(len(self.aj), 0, -1):	# For every hidden layer. (shitty indexing sorry)
       		if self.use_improved_sigmoid:
       			aj_derivative = (2.0/3.0) * (1.7159 - (1 / 1.7159) * self.aj[(i-1)]**2)
       		else:
       			aj_derivative = self.aj[(i-1)] * (1 - self.aj[(i-1)])
       		delta = aj_derivative * np.dot(delta, self.ws[i].T)
       		if i == 1:
       			self.grads[(i-1)] = np.dot(delta.T, X).T / (outputs.shape[0]) #Should have added input X as first element of aj or zj..
       		else:
        		self.grads[(i-1)] = np.dot(delta.T, self.aj[(i-2)]).T / (outputs.shape[0])

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    encode = np.array(Y).reshape(-1)

    return np.eye(num_classes)[encode]


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
