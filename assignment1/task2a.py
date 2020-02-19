import numpy as np
import utils
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] in the range (0, 1)
    """
    bias = np.ones((X.shape[0],1))

    X = X/255

    X = np.append(X, bias, axis = 1)

    assert X.shape[1] == 785,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, 1]
        outputs: outputs of model of shape: [batch size, 1]
    Returns:
        Cross entropy error (float)
    """
    N = targets.shape[0]
    K = targets.shape[1] # = 1

    loss = -1/(N*K) * np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return loss


class BinaryModel:

    def __init__(self, l2_reg_lambda: float):
        # Define number of input nodes
        self.I = 785
        self.w = np.zeros((self.I, 1))
        self.grad = None

        # Hyperparameter for task 3
        self.l2_reg_lambda = l2_reg_lambda
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]-np.sum(targets*np.log(outputs)) / (N*K)
        Returns:
            y: output of model with shape [batch size, 1]
        """
        # Sigmoid

        y = np.zeros((X.shape[0],1))

        for index, im in enumerate(X):
        	im = np.array(im)
        	WT_X = np.dot(self.w.T, im)
        	np.put(y, index, 1/(1 + np.exp(-WT_X)))

        return y

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, 1]
            targets: labels/targets of each image of shape: [batch size, 1]
        """

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        self.grad = np.zeros_like(self.w)
        assert self.grad.shape == self.w.shape,\
            f"Grad shape: {self.grad.shape}, w: {self.w.shape}"

            #Normalized (N*K)

        self.grad = -np.dot((targets - outputs).T,X).T / (targets.shape[0]*targets.shape[1]) # Original 2a

        self.grad = self.grad + np.dot(2 * self.l2_reg_lambda, self.w)	# With regularization. Not sure if correct


    def update_weights(self, learning_rate: float, batch_size: int) -> None:
    	"""
		Funnction that updates the weights of the network with a given batch size 
		and learning rate ( not really needed!)
    	"""
    	self.w = np.add(np.multiply((-learning_rate*(1/batch_size)), self.grad), self.w)

    def zero_grad(self) -> None:
        self.grad = None


def gradient_approximation_test(model: BinaryModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    w_orig = model.w.copy()
    epsilon = 1e-2
    for i in range(w_orig.shape[0]):
        orig = model.w[i].copy()
        model.w[i] = orig + epsilon
        logits = model.forward(X)
        cost1 = cross_entropy_loss(Y, logits)
        model.w[i] = orig - epsilon
        logits = model.forward(X)
        cost2 = cross_entropy_loss(Y, logits)
        gradient_approximation = (cost1 - cost2) / (2 * epsilon)
        model.w[i] = orig
        # Actual gradient
        logits = model.forward(X)
        model.backward(X, logits, Y)
        difference = gradient_approximation - model.grad[i, 0]
        assert abs(difference) <= epsilon**2,\
            f"Calculated gradient is incorrect. " \
            f"Approximation: {gradient_approximation}, actual gradient: {model.grad[i, 0]}\n" \
            f"If this test fails there could be errors in your cross entropy loss function, " \
            f"forward function or backward function"


if __name__ == "__main__":
    category1, category2 = 2, 3
    X_train, Y_train, *_ = utils.load_binary_dataset(category1, category2, 0.1)
    X_train = pre_process_images(X_train)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Simple test for forward pass. Note that this does not cover all errors!
    model = BinaryModel(0.0)
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), .5,
        err_msg="Since the weights are all 0's, the sigmoid activation should be 0.5")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for i in range(2):
        gradient_approximation_test(model, X_train, Y_train)
        model.w = np.random.randn(*model.w.shape)
