import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
np.random.seed(0)

def shuffle_set(X: np.ndarray, Y: np.ndarray):
    index = np.arange(0, X.shape[0])
    np.random.shuffle(index)
    X_shuffled = X[index]
    Y_shuffled = Y[index]

    return X_shuffled, Y_shuffled

def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    output = model.forward(X).argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (output == targets).mean()


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}

    dw1 = 0
    dw2 = 0

    global_step = 0
    for epoch in range(num_epochs):
        if use_shuffle:
            X_train, Y_train = shuffle_set(X_train, Y_train)      # 3a)
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            output = model.forward(X_batch)
            model.backward(X_batch, output , Y_batch)
            if use_momentum:
                dw1 = np.add(learning_rate * model.grads[0], momentum_gamma * dw1)
                dw2 = np.add(learning_rate * model.grads[1], momentum_gamma * dw2)
                model.ws[0] = np.add(model.ws[0], -dw1)
                model.ws[1] = np.add(model.ws[1], -dw2)
            else:
                model.ws[0] = np.add(model.ws[0], -learning_rate * model.grads[0])
                model.ws[1] = np.add(model.ws[1], -learning_rate * model.grads[1])

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss

                _train_loss = cross_entropy_loss(Y_train, model.forward(X_train))
                train_loss[global_step] = _train_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)
                """
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
                """
            global_step += 1
    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)

    X_train= pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    X_test = pre_process_images(X_test)

    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    Y_test = one_hot_encode(Y_test, 10)

    """
    # This code is just here to find the mean and standard deviation of the
    # training set. The values are written down in the cross_entropy function
    # in task2a.py as default values for input parameters. Makes testing easier.
    mean_train = np.mean(X_train)
    print(mean_train)
    std_train = np.std(X_train) #nasty trains
    print(std_train)
    """

    # Hyperparameters
    num_epochs = 20
    learning_rate = .008
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Settings for task 3. Keep all to false for task 2.
    use_shuffle = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        momentum_gamma=momentum_gamma)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model))

    model2 = SoftmaxModel(
        [60, 54 ,10],
        True,
        True)
    model2, train_loss2, val_loss2, train_accuracy2, val_accuracy2 = train(
        model2,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=0.008,
        batch_size=batch_size,
        use_shuffle=True,
        use_momentum=True,
        momentum_gamma=momentum_gamma)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model2.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model2.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model2.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model2))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model2))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model2))

    # Plot loss
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, .4])
    utils.plot_loss(train_loss, "Training Loss with one hidden layer")
    utils.plot_loss(val_loss, "Validation Loss with one hidden layer", "y-")
    utils.plot_loss(train_loss2, "Training Loss with two hidden layers", "b--")
    utils.plot_loss(val_loss2, "Validation Loss with two hidden layers", "y--")
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    # Plot accuracy
    plt.ylim([0.875, 1.0])
    utils.plot_loss(train_accuracy, "Training Accuracy with one hidden layer")
    utils.plot_loss(val_accuracy, "Validation Accuracy with one hidden layer", "y-")
    utils.plot_loss(train_accuracy2, "Training Accuracy with two hidden layers", "b--")
    utils.plot_loss(val_accuracy2, "Validation Accuracy with two hidden layers", "y--")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.savefig("softmax_train_graph_132units.png")
    plt.show()
