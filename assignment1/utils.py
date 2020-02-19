import mnist
import numpy as np
import matplotlib.pyplot as plt


def binary_prune_dataset(class1: int, class2: int,
                         X: np.ndarray, Y: np.ndarray):
    """
    Splits the dataset into the class 1 and class2. All other classes are removed.
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        Y: labels of shape [batch size]
    """

    mask1 = (Y == class1)
    mask2 = (Y == class2)
    mask_total = np.bitwise_or(mask1, mask2)
    Y_binary = Y.copy()
    Y_binary[mask1] = 1
    Y_binary[mask2] = 0
    return X[mask_total], Y_binary[mask_total]


def train_val_split(X: np.ndarray, Y: np.ndarray, val_percentage: float):
    """
    Randomly splits the training dataset into a training and validation set.
    """
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    train_size = int(X.shape[0] * (1 - val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]

    return X_train, Y_train, X_val, Y_val


def load_binary_dataset(class1: int, class2: int, val_percentage: float):
    """
    Loads, prunes and splits the dataset into train, validation and test.
    """
    train_size = 20000
    test_size = 2000
    X_train, Y_train, X_test, Y_test = mnist.load()

    # First 20000 images from train set
    X_train, Y_train = X_train[:train_size], Y_train[:train_size]
    # Last 2000 images from test set
    X_test, Y_test = X_test[-test_size:], Y_test[-test_size:]
    X_train, Y_train = binary_prune_dataset(
        class1, class2, X_train, Y_train
    )
    X_test, Y_test = binary_prune_dataset(
        class1, class2, X_test, Y_test
    )
    # Reshape to (N, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    X_train, Y_train, X_val, Y_val = train_val_split(
        X_train, Y_train, val_percentage
    )
    print(f"Train shape: X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"Validation shape: X: {X_val.shape}, Y: {Y_val.shape}")
    print(f"Test shape: X: {X_test.shape}, Y: {Y_test.shape}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_full_mnist(val_percentage: float):
    """
    Loads and splits the dataset into train, validation and test.
    """
    train_size = 20000
    test_size = 2000
    X_train, Y_train, X_test, Y_test = mnist.load()
    
    # First 20000 images from train set
    X_train, Y_train = X_train[:train_size], Y_train[:train_size]
    # Last 2000 images from test set
    X_test, Y_test = X_test[-test_size:], Y_test[-test_size:]
    # Reshape to (N, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    X_train, Y_train, X_val, Y_val = train_val_split(
        X_train, Y_train, val_percentage
    )
    print(f"Train shape: X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"Validation shape: X: {X_val.shape}, Y: {Y_val.shape}")
    print(f"Test shape: X: {X_test.shape}, Y: {Y_test.shape}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def plot_loss(loss_dict: dict, label: str = None):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
    """
    global_steps = list(loss_dict.keys())
    loss = list(loss_dict.values())
    plt.plot(global_steps, loss, label=label)
