        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
       	delta_k = -(targets - outputs)
       	self.grads[1] = np.dot(delta_k.T, self.aj).T / (outputs.shape[0])

       	if self.use_improved_sigmoid:
       		aj_derivative = (2.0/3.0) * (1.7159 - (1 / 1.7159) * self.aj**2)
       	else:
       		aj_derivative = self.aj * (1 - self.aj)

       	delta_j = aj_derivative * np.dot(delta_k, self.ws[1].T)
        self.grads[0] = np.dot(delta_j.T, X).T / (outputs.shape[0])

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
		"""