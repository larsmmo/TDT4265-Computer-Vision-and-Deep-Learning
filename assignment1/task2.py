import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, BinaryModel, pre_process_images
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: BinaryModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 1]
        model: model of class BinaryModel
    Returns:
        Accuracy (float)
    """
    # Task 2c
    output = model.forward(X)
    output = np.where(output >= 0.5, 1, 0*output)
    correct = np.count_nonzero(targets == output)

    accuracy = correct / len(output)

    return accuracy


def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float # Task 3 hyperparameter. Can be ignored before this.
        ):
    """
        Function that implements logistic regression through mini-batch
        gradient descent for the given hyperparameters
    """
    global X_train, X_val, X_test

    X_train= pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    X_test = pre_process_images(X_test)

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    model = BinaryModel(l2_reg_lambda)

    val_loss_min = 2**31
    weights_min = model.w
    val_loss_prev = 2**31

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            # Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            output = model.forward(X_batch)
            model.backward(X_batch, output, Y_batch)
            model.update_weights(learning_rate, batch_size)

            # Track training loss continuously
            _train_loss = cross_entropy_loss(Y_batch, output)
            train_loss[global_step] = _train_loss
            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss
                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

                # Early stopping. Ends if validation loss increases consistently after passing
                # 20% of train dataset 5 times
                if(_val_loss > val_loss_min):
                    if(_val_loss > val_loss_prev):
                        incr_loss += 1
                    else:
                        incr_loss = 0
                        val_loss_prev = _val_loss 
                    if (incr_loss >= 5):
                        model.w = weights_min
                        print("Early stopping after epoch no:")
                        print(epoch)
                        return model, train_loss, val_loss, train_accuracy, val_accuracy
                else:
                    incr_loss = 0
                    val_loss_min = _val_loss
                    weights_min = model.w

                val_loss_prev = _val_loss

            global_step += 1
    return model, train_loss, val_loss, train_accuracy, val_accuracy


# Load dataset
category1, category2 = 2, 3
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_binary_dataset(
    category1, category2, validation_percentage)

# hyperparameters
num_epochs = 50     # Switched back to 50
learning_rate = 0.2
batch_size = 128
l2_reg_lambda = 0.001

model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=l2_reg_lambda)


print("Final Train Cross Entropy Loss:",
      cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final  Test Entropy Loss:",
      cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Validation Cross Entropy Loss:",
      cross_entropy_loss(Y_val, model.forward(X_val)))


print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Test accuracy:", calculate_accuracy(X_test, Y_test, model))


# Plot loss
#plt.ylim([0., .4]) 
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.legend()
plt.savefig("binary_train_loss.png")
plt.show()


# Plot accuracy
#plt.ylim([0.93, .99])
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.legend()
plt.savefig("binary_train_accuracy.png")
plt.show()
