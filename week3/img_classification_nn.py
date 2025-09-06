import numpy as np
import pickle
from tqdm import tqdm

def my_cl_acc(pred, gt):
    """Vectorized accuracy calculation."""
    pred = np.ravel(pred)
    gt = np.ravel(gt)
    return 100 * np.mean(pred == gt)


def my_1nn_batch(x_train, y_train, x_test, batch_size=200):
    """
    Fully vectorized 1-NN classifier with batching to save memory.

    x_train: (N_train, 28,28) or (N_train, 784)
    y_train: (N_train,)
    x_test:  (N_test, 28,28) or (N_test, 784)
    """
    # Flatten images
    x_train_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test_flat  = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    y_train = np.ravel(y_train)

    N_test = x_test_flat.shape[0]
    y_pred = np.empty(N_test, dtype=y_train.dtype)

    # Split test set into batches
    batches = np.array_split(np.arange(N_test), max(1, N_test // batch_size))

    for batch_indices in tqdm(batches, desc="1-NN batches"):
        batch = x_test_flat[batch_indices]  # shape (b, 784)

        # Squared Euclidean distance: ||x - t||^2 = ||x||^2 + ||t||^2 - 2 x·t
        dists_sq = (
            np.sum(batch**2, axis=1)[:, None] +       # shape (b,1)
            np.sum(x_train_flat**2, axis=1)[None, :] - # shape (1, N_train)
            2 * batch.dot(x_train_flat.T)             # shape (b, N_train)
        )

        # Index of nearest neighbor for each test sample in batch
        nearest_idx = np.argmin(dists_sq, axis=1)
        y_pred[batch_indices] = y_train[nearest_idx]

    return y_pred


def main():
    # Load Fashion-MNIST data
    with open('mnist_fashion.pkl', 'rb') as f:
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    # Run 1-NN
    y_pred = my_1nn_batch(x_train, y_train, x_test, batch_size=200)

    # Compute accuracy
    acc = my_cl_acc(y_pred, y_test)
    print(f"1-NN classification accuracy: {acc:.2f}%")

    # Save predictions
    np.savetxt('PRED_mnist_fashion.dat', y_pred, fmt='%d')


if __name__ == '__main__':
    main()
